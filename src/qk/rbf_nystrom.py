import math
import torch


class RBFNystrom:
    """
    Fast classical Nyström features with an RBF kernel.
      k(a,b) = exp(-||a-b||^2 / (2σ^2))

    - CPU-internal precompute of K(S,S) and its inverse sqrt.
    - Returns ψ(x) on the same device/dtype as x.
    - σ chosen by median heuristic if not provided.
    """
    def __init__(self, landmarks: torch.Tensor, sigma: float = None):
        self.S = landmarks.detach().to(dtype=torch.float32, device="cpu")
        self.M, self.d = int(self.S.shape[0]), int(self.S.shape[1])

        if sigma is None:
            # median heuristic on a subset
            with torch.no_grad():
                idx = torch.randperm(self.M)[:min(512, self.M)]
                A = self.S[idx]
                a2 = (A**2).sum(1, keepdim=True)
                d2 = a2 - 2 * A @ A.T + a2.T
                med = d2.flatten().kthvalue(max(1, d2.numel() // 2)).values.item()
                sigma = math.sqrt(max(med, 1e-6))
        self.sigma2 = float(sigma * sigma)

        with torch.no_grad():
            self.Kss = self._rbf(self.S, self.S)
            e, V = torch.linalg.eigh(self.Kss + 1e-6 * torch.eye(self.M, dtype=torch.float32))
            self.Kss_inv_sqrt = (V * e.clamp_min(1e-6).rsqrt()) @ V.T

    @torch.no_grad()
    def _rbf(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a2 = (A**2).sum(1, keepdim=True)
        b2 = (B**2).sum(1, keepdim=True)
        d2 = a2 - 2 * A @ B.T + b2.T
        return torch.exp(-0.5 * d2 / self.sigma2)

    @torch.no_grad()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        Xc = X.detach().to(dtype=torch.float32, device="cpu")
        KxS = self._rbf(Xc, self.S)             # (N,M) CPU
        psi = KxS @ self.Kss_inv_sqrt           # (N,M) CPU
        return psi.to(device=X.device, dtype=X.dtype)
