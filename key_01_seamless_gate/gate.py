import torch
import torch.nn.functional as F

def seamless_gate(v1: torch.Tensor, v2: torch.Tensor, tau: float = 0.3):
    """The door that no man can close — Rev 3:8 as code."""
    cos = F.cosine_similarity(v1, v2, dim=-1)
    score = torch.sigmoid(cos / tau)                     # 0.0 → 1.0
    fused = score.unsqueeze(-1) * v1 + (1 - score).unsqueeze(-1) * v2
    return fused, score

# === 7-line test — run this file directly ===
if __name__ == "__main__":
    a = torch.randn(1, 768)
    b = a + torch.randn(1, 768)*0.05   # almost identical
    c = torch.randn(1, 768)            # unrelated

    fused_ab, score_ab = seamless_gate(a, b)
    fused_ac, score_ac = seamless_gate(a, c)

    print("Aligned score   :", score_ab.item())   # ~0.99
    print("Unaligned score :", score_ac.item())   # ~0.4–0.6
