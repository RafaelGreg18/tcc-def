import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x0_5, resnext50_32x4d, mobilenet_v2

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum

# ========== CONSTANTS ========== #
N_MELS = 80  # Whisper/Log-Mel comum

# ========== Models ========== #
class SimpleCNN(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()

        fc_input_size = self.calculate_fc_input_size(input_shape)

        self.fc1 = nn.Linear(fc_input_size, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def calculate_fc_input_size(self, input_shape):
        # input_size: (channels, height, width)
        # Forward pass through conv and pool layers to compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)  # Create a dummy tensor
            cnn_layers = nn.Sequential(
                nn.Conv2d(input_shape[0], 6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(6, 16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten()
            )
            output = cnn_layers(dummy_input)  # Forward pass to check output shape

        return output.view(1, -1).shape[1]  # Get flattened size

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# --------------------------
# KWT model
# --------------------------

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, pre_norm=True, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                P_Norm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                P_Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class KWT(nn.Module):
    def __init__(self, input_res, patch_res, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0., pre_norm=True, **kwargs):
        super().__init__()

        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_res[0], p2=patch_res[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def kwt_from_name(model_name: str):
    models = {
        "kwt-1": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 256,
            "dim": 64,
            "heads": 1,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-2": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 512,
            "dim": 128,
            "heads": 2,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-3": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 768,
            "dim": 192,
            "heads": 3,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        }
    }

    assert model_name in models.keys(), f"Unsupported model_name {model_name}; must be one of {list(models.keys())}"

    return KWT(**models[model_name])

# --------------------------
# LSTM model
# --------------------------
class StackedLSTM(nn.Module):
    """StackedLSTM architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fully_ = nn.Linear(256, 80)

    def forward(self, text):
        """Forward pass of the StackedLSTM.

        Parameters
        ----------
        text : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        embedded = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fully_(lstm_out[:, -1, :])
        return final_output

# ------- Blocos básicos -------
class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.gn = nn.GroupNorm(1, out_ch)  # 1 grupo = “LayerNorm por canal”, estável em FL
        self.act = nn.SiLU(inplace=True)  # ou nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.same_ch = (in_ch == out_ch)
        self.conv1 = ConvGNAct(in_ch, out_ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.GroupNorm(1, out_ch),
        )
        self.skip = nn.Identity() if self.same_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.GroupNorm(1, out_ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        x = self.skip(x)
        return self.act(x + y)


class TemporalAttention(nn.Module):
    """Atenção simples sobre a sequência (dim T)."""

    def __init__(self, d_model: int, d_attn: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_attn),
            nn.Tanh(),
        )
        self.v = nn.Linear(d_attn, 1, bias=False)

    def forward(self, x):  # x: [B, T, D]
        a = self.v(self.proj(x)).squeeze(-1)  # [B, T]
        w = torch.softmax(a, dim=1)  # [B, T]
        z = (x * w.unsqueeze(-1)).sum(dim=1)  # [B, D]
        return z, w


# ------- Modelo principal -------
class CRNNResNetAttn(nn.Module):
    """
    Entrada:  x ∈ [B, 1, N_MELS, T]  (use N_MELS=80)
    Saída:    [B, num_classes]
    """

    def __init__(
            self,
            num_classes: int,
            base: int = 64,
            lstm_hidden: int = 256,
            lstm_layers: int = 2,
            attn_dim: int = 128,
            dropout: float = 0.1,
            n_mels: int = N_MELS,  # <<< novo
    ):
        super().__init__()
        C1, C2, C3 = base, base * 2, base * 4  # 64, 128, 256
        self.freq_after = max(1, n_mels // 4)  # <<< 80 // 4 = 20

        # CNN residual
        self.stem = ConvGNAct(1, C1, 3, 1, 1)
        self.layer1 = nn.Sequential(ResBlock(C1, C1), ResBlock(C1, C1))
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Sequential(ResBlock(C1, C2), ResBlock(C2, C2))
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = nn.Sequential(ResBlock(C2, C3), ResBlock(C3, C3))
        self.drop2d = nn.Dropout2d(p=dropout)

        # LSTM: input_size = C3 * freq_after (dinâmico)
        lstm_in = C3 * self.freq_after
        self.rnn = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = TemporalAttention(d_model=2 * lstm_hidden, d_attn=attn_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 2 * lstm_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2 * lstm_hidden, num_classes),
        )

    def forward(self, x):  # x: [B,1,N_MELS,T]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.drop2d(x)
        B, C, Fp, Tp = x.shape

        # Se Fp não bater (por n_mels diferente, trims, etc.), ajusta:
        if Fp != self.freq_after:
            # mantém T', ajusta só a frequência pra freq_after
            x = nn.functional.adaptive_avg_pool2d(x, (self.freq_after, Tp))
            B, C, Fp, Tp = x.shape  # agora Fp == self.freq_after

        # [B, C, F', T'] -> [B, T', C*F']
        x = x.permute(0, 3, 1, 2).contiguous().view(B, Tp, C * Fp)

        # Evita warning de contiguidade dos pesos da LSTM
        self.rnn.flatten_parameters()

        x, _ = self.rnn(x)  # [B, T', 2*hidden]
        z, _ = self.attn(x)  # [B, 2*hidden]
        return self.classifier(z)

# ========== Aux functions ========== #
def get_valid_num_groups(num_channels, max_groups=32):
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1  # fallback to InstanceNorm-like behavior


def convert_bn_to_gn(module, max_groups=32):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = get_valid_num_groups(num_channels, max_groups)
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))
        else:
            convert_bn_to_gn(child, max_groups)


# ========== Factory ========== #
class ModelFactory:
    @staticmethod
    def create(model_name, **kwargs):
        if model_name == 'simplecnn':
            return SimpleCNN(kwargs['input_shape'], int(kwargs['num_classes']))
        elif model_name == 'Mobilenet_v2':
            model = mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, int(kwargs['num_classes']))
            return model
        elif model_name == "Shufflenet_v2_x0_5":
            model = shufflenet_v2_x0_5(weights=None)
            model.fc = nn.Linear(model.fc.in_features, int(kwargs['num_classes']))
            return model
        elif model_name == "Resnext50_32x4d":
            model = resnext50_32x4d(weights=None)
            model.fc = nn.Linear(model.fc.in_features, int(kwargs['num_classes']))
            return model
        elif model_name == "Lstm":
            model = StackedLSTM()
            return model
        elif model_name == "CRNNResNetAttn":
            model = CRNNResNetAttn(num_classes=int(kwargs['num_classes']), base=64, lstm_hidden=256, lstm_layers=2, attn_dim=128,
                                   dropout=0.1, )
            return model
        elif model_name == "KWT":
            model = kwt_from_name("kwt-1")
            in_f = model.mlp_head[-1].in_features
            if getattr(model.mlp_head[-1], "out_features", None) != int(kwargs['num_classes']):
                model.mlp_head[-1] = nn.Linear(in_f, int(kwargs['num_classes']))

            return model
        else:
            raise ValueError(f"Modelo desconhecido: {model_name}")
