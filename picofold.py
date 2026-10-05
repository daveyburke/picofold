"""
picofold.py: a minimal protein structure predictor written in ~200 lines of readable PyTorch. AlphaFold 2 (Nobel Prize
2024) predicted structures with MSAs, pair representations and triangle updates; AlphaFold 3 swapped the structure
module for diffusion. SimpleFold (2025) showed a transformer trained with flow matching (cousin of diffusion) works
almost as well, without needing AlphaFold's special machinery. Toy task: given 16 amino acids, generate the 3D positions
of their 16 C-alpha atoms using flow matching. Trained on ~47k 16-residue fragments from CATH S40 / Protein Data Bank.
@daveyburke - see README at https://github.com/daveyburke/picofold
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42); random.seed(42)
device = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Hyperparameters
L = 16               # residues per window (must match the data file)
D_MODEL = 128        # width of the transformer
N_LAYER = 4          # depth of the transformer
N_HEAD = 4           # attention heads per layer
BATCH = 256          # batch size
TRAIN_STEPS = 10000  # training steps
LR = 1e-3            # learning rate, 1e-3 is a good default
SCALE = 16.0         # divide coordinates (Angstroms) by this so they're roughly unit size
SAMPLE_STEPS = 200   # steps when generating a structure
TAU = 0.3            # fresh noise factor during sampling, 0.3 is a good default, helps with stability
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY" # amino acid tokenization

def attention(q, k, v): # each (B, L, H, hd)
    q, k, v = (z.transpose(1, 2) for z in (q, k, v))             # (B, H, L, hd) one attention problem per head
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])  # (B, H, L, L) every residue pair, scaled
    weights = torch.softmax(scores, dim=-1)                      # (B, H, L, L) each row sums to 1
    return (weights @ v).transpose(1, 2).flatten(2)              # (B, L, d) heads back in place, concatenated

def rope(x, cos, sin):
    x1, x2 = x.chunk(2, dim=-1)  # split half implementation; each (B, L, H, hd/2)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # (B, L, H, hd)

def normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]

def random_rotation_matrix():
    x = normalize([random.gauss(0, 1) for _ in range(3)]) # new x axis: a random direction
    y = [random.gauss(0, 1) for _ in range(3)]
    d = sum(a * b for a, b in zip(x, y))              # dot product: how much of y points along x
    y = normalize([b - d * a for a, b in zip(x, y)])  # new y axis: remove that part
    z = [x[1]*y[2] - x[2]*y[1], x[2]*y[0] - x[0]*y[2], x[0]*y[1] - x[1]*y[0]]  # new z axis: x y cross product
    return [[x[i], y[i], z[i]] for i in range(3)]  # transpose x, y, z to columns of 3x3 matrix

def timestep_embedding(t, dim):
    # Turn scalar t into sines and cosines at many frequencies, to more easily tell nearby noise levels apart
    half = dim // 2
    freqs = 10000.0 ** (-torch.arange(half, device=t.device) / half)  # (d/2,) from 1 down to ~0.0001
    angles = 1000 * t[:, None] * freqs[None, :]  # (B, d/2); the 1000 scales t into the range these frequencies suit
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, d)

### Model: a transformer that maps (sequence, noisy coords, t) -> velocity ###
# Dims: B = batch, L = amino_acid_seq_size, d = D_MODEL, H = N_HEAD, hd = head_dim
class PicoFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.aa_embed = nn.Embedding(len(AMINO_ACIDS), D_MODEL)
        self.coord_in = nn.Linear(3, D_MODEL)
        self.time_mlp = nn.Sequential(nn.Linear(D_MODEL, D_MODEL), nn.SiLU(), nn.Linear(D_MODEL, D_MODEL))

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(D_MODEL),
                'qkv': nn.Linear(D_MODEL, 3 * D_MODEL, bias=False),
                'q_norm': nn.RMSNorm(D_MODEL // N_HEAD),  # QK-norm: keeps attention logits from blowing up
                'k_norm': nn.RMSNorm(D_MODEL // N_HEAD),
                'proj': nn.Linear(D_MODEL, D_MODEL),
                'norm2': nn.LayerNorm(D_MODEL),
                'gate': nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
                'up': nn.Linear(D_MODEL, 4 * D_MODEL, bias=False),
                'down': nn.Linear(4 * D_MODEL, D_MODEL, bias=False),
            }) for _ in range(N_LAYER)
        ])

        self.final_norm = nn.LayerNorm(D_MODEL, elementwise_affine=False)  # no affine part (won't undo the zero init)
        self.out = nn.Linear(D_MODEL, 3)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start out predicting zero velocity
        self.head_dim = D_MODEL // N_HEAD

        # Precompute RoPE angle for each (position, frequency) pair
        theta = 1.0 / (100.0 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))  # theta_i = 1 / 100^2i/d, fast to slow
        angles = torch.arange(L)[:, None] * theta[None, :]  #  m * theta_i (L, hd/2)
        self.register_buffer('cos', angles.cos()[None, :, None, :])  # (1, L, 1, hd/2)
        self.register_buffer('sin', angles.sin()[None, :, None, :])  # (1, L, 1, hd/2)

    def forward(self, seq, x_t, t): # seq: amino-acid ids (B, L); x_t: noisy coords (B, L, 3); t: noise level (B,)
        # 1. Each residue's token is the sum of its amino acid embedding, its noisy coordinates, and timestep embedding
        t = self.time_mlp(timestep_embedding(t, D_MODEL))  # sinusoidal frequencies -> MLP (B, d)
        h = self.aa_embed(seq) + self.coord_in(x_t) + t[:, None, :]  # (B, L, d)

        # 2. Transformer blocks
        B = h.shape[0]
        for block in self.blocks:
            # a) Multi-head attention, with QK-norm and RoPE. Bidirectional (no causal mask: every residue each other)
            q, k, v = block['qkv'](block['norm1'](h)).view(B, L, 3, N_HEAD, self.head_dim).unbind(2)  # each (B, L, H, hd)
            q, k = rope(block['q_norm'](q), self.cos, self.sin), rope(block['k_norm'](k), self.cos, self.sin)
            h = h + block['proj'](attention(q, k, v))  # (B, L, d)

            # b) SwiGLU feed-forward network
            n = block['norm2'](h)  # (B, L, d)
            h = h + block['down'](F.silu(block['gate'](n)) * block['up'](n))  # (B, L, 4d) inside, (B, L, d) out

        return self.out(self.final_norm(h))  # (B, L, 3) velocity

model = PicoFold().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

### Training ###
def load(path):
    rows = [line.split() for line in open(path)] # each line is "domain_id SEQUENCE x1 y1 z1 ... x16 y16 z16"
    seqs = [[AMINO_ACIDS.index(a) for a in r[1]] for r in rows]
    coords = [[float(v) for v in r[2:]] for r in rows] # (N, L*3)
    return torch.tensor(seqs), torch.tensor(coords).view(-1, L, 3)  # (N, L), (N, L, 3)

def train(): # flow matching training
    train_seq, train_xyz = load('train_data.txt')
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(TRAIN_STEPS):
        for g in optimizer.param_groups:
            g['lr'] = LR * min(1.0, (step + 1) / 200) * (1 - step / TRAIN_STEPS)  # warmup, then linear decay

        # A batch of true structures, each in a random orientation, so the model learns rotation invariance
        idx = torch.randint(0, len(train_seq), (BATCH,))                     # (B,)
        seq = train_seq[idx].to(device)                                      # (B, L)
        R = torch.tensor([random_rotation_matrix() for _ in range(BATCH)])   # (B, 3, 3)
        x1 = (train_xyz[idx] @ R).to(device) / SCALE                         # (B, L, 3)

        # Noise level t, skewed near 1 (median 0.7): fine detail matters most
        t = torch.sigmoid(torch.randn(BATCH, device=device) * 1.7 + 0.8)     # (B,)

        # The straight line from noise x0 to structure x1
        x0 = torch.randn_like(x1)                                            # (B, L, 3)
        x_t = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0            # (B, L, 3)

        # Predict the velocity along that line; the loss is just squared error
        loss = F.mse_loss(model(seq, x_t, t), x1 - x0)  # both (B, L, 3); loss is a scalar

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0 or step == TRAIN_STEPS - 1:
            print(f"step {step:5d} | loss {loss.item():.4f}")

    torch.save(model.state_dict(), 'picofold.pt')

### Generation: Euler-Maruyama sampler ###
@torch.no_grad()
def generate(seq): # seq: (B, L) amino-acid ids. Returns (B, L, 3) coordinates in Angstroms.
    x = torch.randn(seq.shape[0], L, 3, device=seq.device)  # (B, L, 3) pure noise
    dt = 1.0 / SAMPLE_STEPS
    for i in range(SAMPLE_STEPS):
        t = i * dt
        x = x - x.mean(dim=1, keepdim=True)          # keep the structure centered
        v = model(seq, x, torch.full((seq.shape[0],), t, device=seq.device))  # (B, L, 3)
        if t < 0.99:
            noise_guess = x - t * v                  # (B, L, 3) the model's guess of the noise in x
            score = -noise_guess / (1 - t)           # (B, L, 3) points toward less noisy structures
            w = (1 - t) / (t + 0.01)                 # scalar correction strength: big early, small late
            # Euler-Maruyama: x += drift*dt + diffusion*sqrt(dt)*randn, drift = v + w*score, diffusion = sqrt(2*w*TAU)
            x = x + (v + w * score) * dt + math.sqrt(2 * dt * w * TAU) * torch.randn_like(x)
        else:
            x = x + v * dt                           # last few steps: plain Euler
    return x * SCALE                                 # (B, L, 3) in Angstroms

### Evaluation ###
LDDT_THRESHOLDS = (0.5, 1.0, 2.0, 4.0) # Angstroms
def lddt(pred, true, cutoff=15.0):
    # For each pair of residues within `cutoff` Angstroms in the true structure,
    # check whether the predicted distance is within each threshold of the true one
    passed, total = 0, 0
    for i in range(len(true)):
        for j in range(i + 1, len(true)):
            d_true = math.dist(true[i], true[j])
            if d_true >= cutoff:
                continue
            err = abs(math.dist(pred[i], pred[j]) - d_true)
            passed += sum(err < thr for thr in LDDT_THRESHOLDS)
            total += len(LDDT_THRESHOLDS)
    return passed / total

def drmsd(pred, true):
    # Root-mean-square error over all pairwise distances (Angstroms). Like RMSD, but no superposition needed.
    sq_errors = [(math.dist(pred[i], pred[j]) - math.dist(true[i], true[j])) ** 2
                 for i in range(len(true)) for j in range(i + 1, len(true))]
    return math.sqrt(sum(sq_errors) / len(sq_errors))

def avg(metric, preds, trues):
    preds, trues = preds.tolist(), trues.tolist()   # (N, L, 3) tensors -> nested lists
    return sum(metric(p, t) for p, t in zip(preds, trues)) / len(preds)

if __name__ == '__main__': # train, generate, and eval on held-out data
    train()
    test_seq, test_xyz = load('test_data.txt')
    n_eval = min(1000, len(test_seq))
    seq, true = test_seq[:n_eval].to(device), test_xyz[:n_eval]  # (N, L), (N, L, 3)
    pred = generate(seq)  # (N, L, 3)
    print(f"\ntest lDDT {avg(lddt, pred, true):.3f} (1 = perfect), dRMSD {avg(drmsd, pred, true):.2f} Angstroms")
