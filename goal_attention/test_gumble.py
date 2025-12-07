import torch
import torch.nn as nn

def gumbel_sigmoid(logits, tau=0.3, hard=True):
    # probs
    probs = torch.sigmoid(logits)

    # gumbel noise
    g = -torch.log(-torch.log(torch.rand_like(probs) + 1e-8) + 1e-8)

    # soft gumbel-sigmoid
    y = torch.sigmoid((torch.log(probs + 1e-8) - torch.log(1 - probs + 1e-8) + g) / tau)

    if hard:
        # hard sample
        y_hard = (y > 0.5).float()
        y_out = y_hard + (y - y_hard).detach()
        return y_out, y, y_hard

    return y, y, None


# ===== TEST =====
logits = torch.tensor([0.0, 2.0, -3.0, 1.5, -1.0])
z, y_soft, y_hard = gumbel_sigmoid(logits, tau=0.1, hard=True)

print("logits: ", logits)
print("y_soft:", y_soft)
print("y_hard:", y_hard)
print("z (output forward):", z)

print("\nCheck if output is binary (0/1):")
print((z == 0) | (z == 1))