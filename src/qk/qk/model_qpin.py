# qk/model_qpin.py
import os
import torch
import torch.nn as nn
from typing import Tuple
from qkernel import QuantumNystromFeatures
from rbf_nystrom import RBFNystrom


class PINN_QK(nn.Module):
    """
    Physics-informed model with hybrid inputs:
      - ψ_q(x): (detached) Nyström features from a quantum (or RBF) kernel
      - x_enc : learned encoder on raw stats x  → enables u_x
      - t     : time/cycle scalar

    u = F([ψ_q(x)_detached, x_enc, t])
    g = G([ψ_q(x)_detached, x_enc, t, u, u_t, u_x])

    Notes
    -----
    - Quantum features are computed on CPU and detached by default (fast).
    - Raw-x path carries gradients so you get u_x via autograd.
    - Set env QK_USE_RBF=1 to switch to classical RBF Nyström features.
    - If you *must* backprop through kernel features, set qk_trainable=True
      (expect large slowdowns with quantum parameter-shift).
    """
    def __init__(
        self,
        landmarks: torch.Tensor,    # (M, d_stats) tensor (stats only, no time)
        M: int,
        alpha: float = 0.7,
        beta: float = 0.2,
        qk_qubits: int = 6,
        qk_depth: int = 1,
        device: str = "cuda",
        hidden: int = 128,
        qk_trainable: bool = False,  # False → detach ψ; True → very slow with quantum
        use_raw_skip: bool = True,   # keep True to enable u_x through x
    ) -> None:
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.alpha, self.beta = float(alpha), float(beta)
        self.M = int(M)
        self.d = int(landmarks.shape[1])
        self.use_raw_skip = bool(use_raw_skip)
        self.qk_trainable = bool(qk_trainable)

        # ---- Featureizer selection (Quantum or RBF) ----
        use_rbf = os.environ.get("QK_USE_RBF", "0") == "1"
        if use_rbf:
            self.feat = RBFNystrom(landmarks)  # CPU-internal, returns ψ on x.device
        else:
            self.feat = QuantumNystromFeatures(landmarks=landmarks, n_qubits=qk_qubits, depth=qk_depth)

        # ---- Raw-x encoder (for u_x) ----
        self.enc_x = nn.Sequential(
            nn.Linear(self.d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )

        # ---- Solution network: u = F([ψ, x_enc, t]) ----
        self.solution_u = nn.Sequential(
            nn.Linear(self.M + hidden + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),             nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        # ---- Dynamics network: g = G([ψ, x_enc, t, u, u_t, u_x]) ----
        # input size: M + hidden + 1 + 1 + 1 + d
        self.dynamics_g = nn.Sequential(
            nn.Linear(self.M + hidden + 1 + 1 + 1 + self.d, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),                               nn.Tanh(),
            nn.Linear(hidden, 1)
        )

        self.to(self.device)

    def forward(self, xt: torch.Tensor):
        """
        xt: (N, d + 1)  → [:, :-1] = x (d stats), [:, -1:] = t (1)

        Returns:
        u   : (N, 1)
        u_t : (N, 1)
        u_x : (N, d)
        g   : (N, 1)
        """
        # move + cast (no requires_grad yet)
        xt = xt.to(self.device).float()
        x  = xt[:, :-1]                # (N, d)
        t  = xt[:, -1:].contiguous()   # (N, 1)

        # ---- kernel features ψ(x) (kept non-trainable by default) ----
        if self.qk_trainable:
            psi = self.feat(x)  # WARNING: slow for quantum; ok for RBF
        else:
            with torch.no_grad():
                psi = self.feat(x)
        psi = psi.to(self.device, dtype=x.dtype)        # (N, M)

        # ---- everything that needs autograd (u, u_t, u_x, g) ----
        with torch.enable_grad():
            # create grad-tracking copies
            x_req = x.detach().clone().requires_grad_(True)     # (N, d)
            t_req = t.detach().clone().requires_grad_(True)     # (N, 1)

            # re-encode x so u depends on x_req → enables u_x
            x_enc = self.enc_x(x_req)                           # (N, hidden)

            # u = F([ψ_detached, x_enc, t_req])
            z_u = torch.cat([psi.detach(), x_enc, t_req], dim=1)  # (N, M+hidden+1)
            u   = self.solution_u(z_u)                             # (N, 1)

            # u_t, u_x via autograd
            ones = torch.ones_like(u)
            u_t  = torch.autograd.grad(u, t_req, grad_outputs=ones,
                                    create_graph=True, retain_graph=True, allow_unused=False)[0]  # (N,1)
            u_x  = torch.autograd.grad(u, x_req, grad_outputs=ones,
                                    create_graph=True, retain_graph=True, allow_unused=False)[0]  # (N,d)

            # g = G([ψ_detached, x_enc, t_req, u, u_t, u_x])
            g_in = torch.cat([psi.detach(), x_enc, t_req, u, u_t, u_x], dim=1)
            g    = self.dynamics_g(g_in)                          # (N,1)

        return u, u_t, u_x, g
