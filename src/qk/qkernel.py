import math, os, time, hashlib, numpy as np
import torch, torch.nn as nn
import pennylane as qml  

def _p(msg: str):
    if os.environ.get("QK_PROGRESS", "0") == "1":
        print(msg, flush=True)

def _inv_sqrt_psd_cpu(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    A = A.to(dtype=torch.float32, device="cpu")
    I = torch.eye(A.shape[-1], dtype=torch.float32, device="cpu")
    e, V = torch.linalg.eigh(A + eps * I)
    e = torch.clamp(e, min=eps).rsqrt()
    return (V * e) @ V.mT

def _fingerprint(t: torch.Tensor) -> str:
    arr = t.to(dtype=torch.float32, device="cpu").contiguous().numpy().tobytes()
    return hashlib.sha1(arr).hexdigest()

def _row_key_np(x: np.ndarray, q: float = 1e-6) -> str:
    # light quantization → stable hash across nearly-identical floats
    xr = np.round(x / q) * q
    return hashlib.sha1(xr.tobytes()).hexdigest()

class QuantumNystromFeatures(nn.Module):
    """
    Fast CPU quantum Nyström features via state caching:
      • Build/caches |ψ(s_m)⟩ for all landmarks S once → S_states (M, 2^n)
      • Optionally cache |ψ(x)⟩ rows on disk+RAM (one file per unique x)
      • K(x,S) = | X_states @ S_states^H |^2 (cheap dense BLAS)
      • K(S,S) from S_states Gram; then ψ(x) = K(x,S) @ KSS^{-1/2}
    """
    def __init__(self, landmarks: torch.Tensor, n_qubits: int = 8, depth: int = 2):
        super().__init__()
        self.S_cpu = landmarks.detach().to(dtype=torch.float32, device="cpu")
        self.M, self.d = int(self.S_cpu.shape[0]), int(self.S_cpu.shape[1])
        self.nq = int(n_qubits)
        self.depth = int(depth)
        self.dim = 1 << self.nq  # 2^n amplitudes

        # simple projection from d -> nq (downsample) or pad up to nq
        if self.d == self.nq:
            self.proj_idx_cpu = None
        elif self.d > self.nq:
            self.proj_idx_cpu = torch.linspace(0, self.d - 1, steps=self.nq).round().long().cpu()
        else:
            self.proj_idx_cpu = None  # we'll pad with zeros

        # choose fastest device available
        try:
            self.dev = qml.device("lightning.qubit", wires=self.nq)
            _p("[QK] Using lightning.qubit")
        except Exception:
            self.dev = qml.device("default.qubit", wires=self.nq)
            _p("[QK] Using default.qubit")

        # feature map (pure-numpy inputs)
        def feature_map_numpy(v: np.ndarray):
            if self.d == self.nq:
                u = v
            elif self.d > self.nq:
                idx = self.proj_idx_cpu.numpy()
                u = v[idx]
            else:
                u = np.concatenate([v, np.zeros(self.nq - self.d, dtype=v.dtype)], axis=0)

            for _ in range(self.depth):
                for w in range(self.nq):
                    # numeric param (no autograd): interface=None is fastest
                    qml.RY(math.pi * np.tanh(float(u[w])), wires=w)
                for w in range(self.nq - 1):
                    qml.CZ(wires=[w, w + 1])
                qml.CZ(wires=[self.nq - 1, 0])

        @qml.qnode(self.dev, interface=None, diff_method=None)
        def state_from_vec(v_np: np.ndarray):
            feature_map_numpy(v_np)
            return qml.state()

        self._state_from_vec = state_from_vec

        # ---- cache roots
        base = os.environ.get("QK_CACHE_DIR", "qkcache")
        os.makedirs(base, exist_ok=True)
        finger_S = _fingerprint(self.S_cpu)
        stem = f"nq{self.nq}_d{self.depth}_M{self.M}_{finger_S}"

        # landmark states cache (M, 2^n) complex64
        self.Sstates_path = os.path.join(base, f"SSTATES_{stem}.pt")
        # K(S,S) cache (float32)
        self.KSS_path = os.path.join(base, f"KSS_{stem}.pt")
        # per-row |ψ(x)⟩ cache directory
        self.Xstates_dir = os.path.join(base, f"XSTATES_{stem}")
        os.makedirs(self.Xstates_dir, exist_ok=True)

        # in-memory memo for x rows (hash → torch.complex64 [dim])
        self._xstate_mem = {}

        # ---- build/load S states
        if os.path.exists(self.Sstates_path):
            _p(f"[QK] Loading landmark states: {self.Sstates_path}")
            Sstates = torch.load(self.Sstates_path, map_location="cpu")
        else:
            _p(f"[QK] Building landmark states ({self.M} × {self.dim}) …")
            t0 = time.time()
            Sstates = torch.empty((self.M, self.dim), dtype=torch.complex64, device="cpu")
            for m in range(self.M):
                if os.environ.get("QK_PROGRESS", "0") == "1":
                    print(f"\r[QK] ψ(S) {m+1}/{self.M}", end="", flush=True)
                v = self.S_cpu[m].numpy()
                psi = self._state_from_vec(v)  # np.complex128 [dim]
                Sstates[m] = torch.from_numpy(np.asarray(psi)).to(torch.complex64)
            if os.environ.get("QK_PROGRESS", "0") == "1":
                print()
            torch.save(Sstates, self.Sstates_path)
            _p(f"[QK] Landmark states done in {time.time()-t0:.1f}s → {self.Sstates_path}")

        self.Sstates = Sstates  # (M, dim), complex64, CPU

        # ---- build/load K(S,S) and its inverse sqrt
        if os.path.exists(self.KSS_path):
            _p(f"[QK] Loading cached K(S,S): {self.KSS_path}")
            KSS = torch.load(self.KSS_path, map_location="cpu")
        else:
            _p("[QK] Computing K(S,S) from cached states …")
            # Gram via batched dot products
            # KSS[i,j] = | <ψ(S_i)|ψ(S_j)> |^2
            psi = Sstates  # (M, dim)
            # inner = psi @ psi^H (complex), shape (M, M)
            inner = psi @ psi.conj().mT
            KSS = inner.abs().pow(2).to(torch.float32)
            torch.save(KSS, self.KSS_path)
            _p(f"[QK] Saved K(S,S) to {self.KSS_path}")

        self.Kss_cpu = KSS.to(dtype=torch.float32, device="cpu")
        self.Kss_inv_sqrt_cpu = _inv_sqrt_psd_cpu(self.Kss_cpu)

    # ------- helpers -------
    def _x_row_state(self, x_row_cpu: torch.Tensor) -> torch.Tensor:
        """
        Return |ψ(x)⟩ (complex64 [dim]) from RAM/disk cache or compute+cache.
        """
        v_np = x_row_cpu.to(dtype=torch.float32, device="cpu").numpy()
        key = _row_key_np(v_np)
        st = self._xstate_mem.get(key, None)

        if st is None:
            fpath = os.path.join(self.Xstates_dir, f"{key}.pt")
            if os.path.exists(fpath):
                try:
                    st = torch.load(fpath, map_location="cpu")
                except Exception:
                    st = None

        if st is None:
            psi = self._state_from_vec(v_np)  # np complex
            st = torch.from_numpy(np.asarray(psi)).to(torch.complex64)
            self._xstate_mem[key] = st
            try:
                torch.save(st, os.path.join(self.Xstates_dir, f"{key}.pt"))
            except Exception:
                pass

        return st  # (dim,) complex64 CPU

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, d) → ψ(x) ≡ K(x,S) KSS^{-1/2} with K(x,S)[n,m]=|<ψ(x_n)|ψ(S_m)>|^2
        Returns on same device/dtype as x.
        """
        x_cpu = x.detach().to(device="cpu", dtype=torch.float32)
        N = int(x_cpu.shape[0])

        show = os.environ.get("QK_PROGRESS", "0") == "1"
        if show: print(f"[QK] Building K(x,S) via state cache for N={N}, M={self.M}", flush=True)

        # Build X_states (N, dim) complex64 by caching per-row states
        Xstates = torch.empty((N, self.dim), dtype=torch.complex64, device="cpu")
        for n in range(N):
            st = self._x_row_state(x_cpu[n])
            Xstates[n] = st
            if show and ((n+1) % max(1, N//10) == 0 or n == N-1):
                print(f"\r[QK] x-states {n+1}/{N}", end="", flush=True)
        if show: print()

        # K(x,S) = | X @ S^H |^2
        inner = Xstates @ self.Sstates.conj().mT  # (N, M) complex
        KxS = inner.abs().pow(2).to(torch.float32)  # (N, M)

        psi = KxS @ self.Kss_inv_sqrt_cpu  # (N, M) float32 on CPU
        return psi.to(device=x.device, dtype=x.dtype)
