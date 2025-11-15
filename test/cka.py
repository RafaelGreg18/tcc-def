# pip install torch torchvision scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ---------- CKA linear (minibatch) ----------
def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def _hsic_linear(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    # Gram lineares
    K = X @ X.T
    L = Y @ Y.T
    Kc = _center_gram(K)
    Lc = _center_gram(L)
    # HSIC não normalizado (versão amostral)
    return (Kc * Lc).sum() / (X.shape[0] - 1) ** 2

@torch.no_grad()
def cka_linear_minibatch(model_a, model_b, dataloader, hook_a, hook_b, device="cpu"):
    model_a.eval(); model_b.eval()
    model_a.to(device); model_b.to(device)

    hsic_xy = hsic_xx = hsic_yy = 0.0
    batches = 0

    for x, *_ in dataloader:
        x = x.to(device)

        Fa = hook_a(model_a, x)   # (B, D_a)
        Fb = hook_b(model_b, x)   # (B, D_b)

        # normalização por feature ajuda a estabilidade numérica
        Fa = (Fa - Fa.mean(0)) / (Fa.std(0) + 1e-6)
        Fb = (Fb - Fb.mean(0)) / (Fb.std(0) + 1e-6)

        hsic_xy += _hsic_linear(Fa, Fb).item()
        hsic_xx += _hsic_linear(Fa, Fa).item()
        hsic_yy += _hsic_linear(Fb, Fb).item()
        batches += 1

    num = hsic_xy / batches
    den = ((hsic_xx / batches) * (hsic_yy / batches)) ** 0.5
    return num / (den + 1e-12)  # CKA em [0,1] (quanto maior, mais parecido)

# ---------- dois modelos e hooks ----------
class MLP(nn.Module):
    def __init__(self, in_dim=28*28, h1=256, h2=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)

def hook_fc1(model, x):
    x = x.view(x.size(0), -1)
    return F.relu(model.fc1(x))

def hook_fc2(model, x):
    x = x.view(x.size(0), -1)
    h1 = F.relu(model.fc1(x))
    return F.relu(model.fc2(h1))

# ---------- dados dummy (sem rótulo) ----------
torch.manual_seed(0)
X = torch.randn(4000, 1, 28, 28)  # 4k imagens sintéticas 28x28
loader = DataLoader(TensorDataset(X, torch.zeros(len(X))), batch_size=256, shuffle=False)

# ---------- instancie dois modelos (semente diferente) ----------
torch.manual_seed(1); model_a = MLP()
torch.manual_seed(2); model_b = MLP()

# ---------- calcule CKA nas camadas fc1 e fc2 ----------
cka_fc1 = cka_linear_minibatch(model_a, model_b, loader, hook_fc1, hook_fc1, device="cpu")
cka_fc2 = cka_linear_minibatch(model_a, model_b, loader, hook_fc2, hook_fc2, device="cpu")

print(f"CKA(fc1 A, fc1 B) = {cka_fc1:.3f}")
print(f"CKA(fc2 A, fc2 B) = {cka_fc2:.3f}")
