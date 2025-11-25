import torch
import torch.nn.functional as F

def seamless_gate(v1: torch.Tensor, v2: torch.Tensor, tau: float = 0.3):
    """The door that no one can shut — Rev 3:8 as code."""
    cos = F.cosine_similarity(v1, v2, dim=-1)
    score = torch.sigmoid(cos / tau)
    fused = score.unsqueeze(-1) * v1 + (1 - score).unsqueeze(-1) * v2
    return fused, score

if __name__ == "__main__":
    a = torch.randn(1, 768)
    b = a + torch.randn(1, 768)*0.05
    c = torch.randn(1, 768)
    _, sab = seamless_gate(a, b)
    _, sac = seamless_gate(a, c)
    print("Aligned →", sab.item())
    print("Unaligned →", sac.item())
