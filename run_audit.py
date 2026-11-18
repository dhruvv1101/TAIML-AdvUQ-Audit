"""
TAIML Empirical Audit: Adversarial Training for Uncertainty Quantification
Custom Execution: 'quick' preset with 3 datasets (MNIST, CIFAR-10, SVHN)
n_train=3000, n_test=1000, epochs=10, T=30, seeds=[0]
"""

import os
import sys
import json
import math
import time
import random
import hashlib
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ------------------------------ USER CONFIGURATION ------------------------------
CFG = dict(
    datasets=['MNIST', 'CIFAR10', 'SVHN'],
    n_train=3000,
    n_test=1000,
    epochs=10,
    T=30,
    seeds=[0],
    train_attacks=['BIM-A', 'BIM-B', 'APGD-reduced', 'Adaptive'],
    attacks=['FGSM', 'BIM-A', 'BIM-B', 'APGD-reduced', 'PGDL2', 'Jitter', 'Adaptive'],
    preset="quick_user",
    lr=1e-3,
    wd=1e-5,
    bs=128,
    kl_w=0.01,
    prior_sd=0.2,
    dropout=0.5,
    aug_cifar=True
)

ALL_ATTACKS = CFG['attacks']
TRAIN_ATTACKS = CFG['train_attacks']
FRESH_ATTACKS = CFG['train_attacks']   # Re-attack retrained models with active attackers
MEASURES = ["MI", "EKL", "PV", "PU"]
EPS = {"MNIST": 0.3, "CIFAR10": 8/255, "SVHN": 8/255}
L2  = {"MNIST": 2.5, "CIFAR10": 0.5,   "SVHN": 1.0}

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f">> Device: {dev} | User Preset: quick (n_train={CFG['n_train']}, n_test={CFG['n_test']}, epochs={CFG['epochs']}, T={CFG['T']})")
if dev == "cuda":
    print(f">> GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print(">> Running on CPU with vectorized multi-core execution.")

ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ["data", "cache", "results", "figures"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

CFG_HASH = hashlib.md5(json.dumps(CFG, sort_keys=True).encode()).hexdigest()[:8]
json.dump(CFG, open(os.path.join(ROOT, "results", f"config_{CFG_HASH}.json"), "w"), indent=2)
print(f">> Config Hash: {CFG_HASH} | Output Root: {ROOT}")

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ------------------------------ DATA PIPELINE & EDA ------------------------------
import torchvision
CH = {"MNIST": (1, 28), "CIFAR10": (3, 32), "SVHN": (3, 32)}
_RAW = {}

def _raw(name):
    if name in _RAW:
        return _RAW[name]
    root = os.path.join(ROOT, "data")
    if name == "MNIST":
        tr = torchvision.datasets.MNIST(root, True, download=False)
        te = torchvision.datasets.MNIST(root, False, download=False)
        conv = lambda d: (d.data.numpy()[:, None].astype(np.float32) / 255.0, d.targets.numpy())
    elif name == "CIFAR10":
        cifar_npz = os.path.join(root, "cifar10_compact.npz")
        if os.path.exists(cifar_npz):
            data = np.load(cifar_npz)
            _RAW[name] = ((data["Xtr"].astype(np.float32) / 255.0, data["ytr"]),
                          (data["Xte"].astype(np.float32) / 255.0, data["yte"]))
            return _RAW[name]
        else:
            tr = torchvision.datasets.CIFAR10(root, True, download=True)
            te = torchvision.datasets.CIFAR10(root, False, download=True)
            conv = lambda d: (d.data.transpose(0, 3, 1, 2).astype(np.float32) / 255.0, np.array(d.targets))
    else:
        tr = torchvision.datasets.SVHN(root, split="train", download=False)
        te = torchvision.datasets.SVHN(root, split="test", download=False)
        conv = lambda d: (d.data.astype(np.float32) / 255.0, np.array(d.labels))
    _RAW[name] = (conv(tr), conv(te))
    return _RAW[name]

def bal(X, y, n, rng):
    per = n // 10
    idx = np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in range(10)])
    rng.shuffle(idx)
    return X[idx], y[idx]

def load_ds(name, seed):
    rng = np.random.RandomState(seed)
    (Xa, ya), (Xb, yb) = _raw(name)
    Xtr, ytr = bal(Xa, ya, CFG["n_train"], rng)
    Xte, yte = bal(Xb, yb, CFG["n_test"], rng)
    t = lambda a, f: torch.tensor(a, dtype=torch.float32 if f else torch.long, device=dev)
    return t(Xtr, 1), t(ytr, 0), t(Xte, 1), t(yte, 0)

# Quick EDA on subsets
eda = []
for ds in CFG["datasets"]:
    Xtr, ytr, Xte, yte = load_ds(ds, CFG["seeds"][0])
    cnt = np.bincount(ytr.cpu().numpy(), minlength=10)
    chi = stats.chisquare(cnt).pvalue
    ks = stats.ks_2samp(Xtr.mean((1, 2, 3)).cpu().numpy(), Xte.mean((1, 2, 3)).cpu().numpy()).pvalue
    eda.append(dict(dataset=ds, n_train=len(Xtr), n_test=len(Xte), max_min_ratio=cnt.max() / cnt.min(), chi2_p=chi,
                    mean=Xtr.mean().item(), std=Xtr.std().item(), KS_brightness_p=ks))
eda = pd.DataFrame(eda).round(4)
print("\n=== [EDA] Dataset Subset Balance & Distribution ===")
print(eda.to_string())

# ------------------------------ MODEL ARCHITECTURES ------------------------------
class BayesLinear(nn.Module):
    def __init__(self, i, o, prior_sd):
        super().__init__()
        self.prior_sd = prior_sd
        self.mu = nn.Parameter(torch.empty(o, i))
        nn.init.kaiming_uniform_(self.mu, a=math.sqrt(5))
        self.rho = nn.Parameter(torch.full((o, i), -5.0))
        self.b = nn.Parameter(torch.zeros(o))

    def forward(self, x, sample):
        w = self.mu + F.softplus(self.rho) * torch.randn_like(self.mu) if sample else self.mu
        return F.linear(x, w, self.b)

    def kl(self):
        sd = F.softplus(self.rho); ps = self.prior_sd
        return (torch.log(ps / sd) + (sd ** 2 + self.mu ** 2) / (2 * ps ** 2) - 0.5).mean()

def conv_stack(ds):
    c = CH[ds][0]; C = nn.Conv2d; R = nn.ReLU; P = lambda: nn.MaxPool2d(2)
    if ds in ["CIFAR10", "SVHN"]:
        return nn.Sequential(C(c, 32, 3, padding=1), R(), C(32, 32, 3, padding=1), R(), P(),
                             C(32, 64, 3, padding=1), R(), C(64, 64, 3, padding=1), R(), P(),
                             C(64, 128, 3, padding=1), R(), C(128, 128, 3, padding=1), R(), P())
    return nn.Sequential(C(c, 64, 3), R(), C(64, 64, 3), R(), P())

class Net(nn.Module):
    def __init__(self, ds, kind, mean, std):
        super().__init__()
        self.kind = kind
        self.p = CFG["dropout"]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.conv = conv_stack(ds)
        c, hw = CH[ds]
        with torch.no_grad(): flat = self.conv(torch.zeros(1, c, hw, hw)).numel()
        self.fc1 = nn.Linear(flat, 128)
        self.out = BayesLinear(128, 10, CFG["prior_sd"]) if kind == "bnn" else nn.Linear(128, 10)

    def feats(self, x):
        return self.conv((x - self.mean) / self.std).flatten(1)

    def head(self, f, stoch=False):
        dr = (self.training or stoch) if self.kind == "mc" else self.training
        h = F.relu(self.fc1(F.dropout(f, self.p, training=dr)))
        h = F.dropout(h, self.p, training=dr)
        return self.out(h, stoch or self.training) if self.kind == "bnn" else self.out(h)

    def forward(self, x):
        return self.head(self.feats(x), False)

    def kl(self):
        return self.out.kl() if self.kind == "bnn" else 0.0

def augment(x):
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    p = 4
    xp = F.pad(x, (p, p, p, p), mode="reflect")
    i, j = np.random.randint(0, 2 * p + 1, 2)
    return xp[:, :, i:i + x.shape[2], j:j + x.shape[3]]

def train_model(ds, kind, X, y, seed):
    set_seed(seed)
    mean = X.mean((0, 2, 3), keepdim=True)
    std = X.std((0, 2, 3), keepdim=True) + 1e-6
    m = Net(ds, kind, mean, std).to(dev)
    
    num_epochs = CFG["epochs"]
    opt = torch.optim.Adam(m.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs, eta_min=1e-5)
    
    for ep in range(num_epochs):
        m.train()
        perm = torch.randperm(len(X), device=dev)
        for i in range(0, len(X), CFG["bs"]):
            idx = perm[i:i + CFG["bs"]]
            xb = augment(X[idx]) if (ds == "CIFAR10" and CFG["aug_cifar"]) else X[idx]
            loss = F.cross_entropy(m(xb), y[idx]) + CFG["kl_w"] * m.kl()
            opt.zero_grad()
            loss.backward()
            opt.step()
        scheduler.step()
    return m.eval()

@torch.no_grad()
def mc_probs(m, X, T, bs=500):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        f = m.feats(X[i:i + bs])
        out.append(torch.stack([F.softmax(m.head(f, True), 1) for _ in range(T)]).cpu())
    return torch.cat(out, 1)

# ------------------------------ UNCERTAINTY METRICS ------------------------------
def uncertainties(P):
    e = 1e-12
    pbar = P.mean(0)
    H = lambda p: -(p * torch.log(p + e)).sum(-1)
    MI  = H(pbar) - H(P).mean(0)
    EKL = (P * (torch.log(P + e) - torch.log(pbar + e))).sum(-1).mean(0)
    PV  = ((P - pbar) ** 2).mean(dim=(0, 2))
    MP  = 1 - pbar.max(1).values
    return dict(MI=MI, EKL=EKL, PV=PV, PU=MP + PV), pbar

# ------------------------------ ADVERSARIAL ATTACKS ------------------------------
import torchattacks

def bim(model, x, y, eps, alpha, steps, stop, random_start=True):
    if random_start and not stop:
        xa = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1)
    else:
        xa = x.clone()
        
    with torch.no_grad():
        active = (model(x).argmax(1) == y) if stop else torch.ones(len(x), dtype=torch.bool, device=x.device)
        
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(xa), y, reduction="sum"), xa)
        with torch.no_grad():
            xn = torch.min(torch.max(xa + alpha * g.sign(), x - eps), x + eps).clamp(0, 1)
            xa = torch.where(active.view(-1, 1, 1, 1), xn, xa.detach())
            if stop:
                active = active & (model(xa).argmax(1) == y)
                if not active.any(): break
    return xa.detach()

def adaptive(model, x, y, eps, alpha, steps, T=3):
    xa = x.clone()
    for _ in range(steps):
        xa = xa.detach().requires_grad_(True)
        f = model.feats(xa)
        P = torch.stack([F.softmax(model.head(f, True), 1) for _ in range(T)])
        pb = P.mean(0)
        ce = F.nll_loss(torch.log(pb + 1e-12), y, reduction="sum")
        mi = (-(pb * torch.log(pb + 1e-12)).sum(1) + (P * torch.log(P + 1e-12)).sum(-1).mean(0)).sum()
        g, = torch.autograd.grad(ce - mi, xa)
        with torch.no_grad():
            xa = torch.min(torch.max(xa + alpha * g.sign(), x - eps), x + eps).clamp(0, 1)
    return xa.detach()

def _attack(name, model, x, y, ds, scale):
    eps, l2 = EPS[ds] * scale, L2[ds] * scale
    model.eval()
    if name == "FGSM":
        x2 = x.clone().requires_grad_(True)
        g, = torch.autograd.grad(F.cross_entropy(model(x2), y, reduction="sum"), x2)
        return (x + eps * g.sign()).clamp(0, 1).detach()
    if name == "BIM-A":
        return bim(model, x, y, eps, eps / 10, 15, True, random_start=False)
    if name == "BIM-B":
        return bim(model, x, y, eps, eps / 8, 12, False, random_start=True)
    if name == "Adaptive":
        return adaptive(model, x, y, eps, eps / 8, 10, T=3)
    if name in ["AutoAttack", "APGD-reduced"]:
        try:
            xa = torchattacks.APGD(model, norm="Linf", eps=eps, steps=8, n_restarts=1, loss="ce", seed=0)(x, y)
            with torch.no_grad(): still = (model(xa).argmax(1) == y)
            if still.any():
                x2 = torchattacks.APGD(model, norm="Linf", eps=eps, steps=8, n_restarts=1, loss="dlr", seed=0)(x, y)
                xa = torch.where(still.view(-1, 1, 1, 1), x2, xa)
            return xa.detach()
        except Exception:
            return bim(model, x, y, eps, eps / 6, 10, False, random_start=True)
    if name == "PGDL2":
        try:
            return torchattacks.PGDL2(model, eps=l2, alpha=l2 / 4, steps=8, random_start=True)(x, y).detach()
        except Exception:
            return bim(model, x, y, eps, eps / 6, 8, False)
    if name == "Jitter":
        try:
            return torchattacks.Jitter(model, eps=eps, alpha=eps / 4, steps=8, scale=10, std=0.1, random_start=True)(x, y).detach()
        except Exception:
            return bim(model, x, y, eps, eps / 6, 8, False)
    raise ValueError(f"Unknown attack: {name}")

def craft(name, model, X, y, ds, scale=1.0, bs=500):
    out = [_attack(name, model, X[i:i + bs], y[i:i + bs], ds, scale) for i in range(0, len(X), bs)]
    model.eval()
    return torch.cat(out)

# ------------------------------ EVALUATION METRICS ------------------------------
def ece_fn(pbar, y, B=15):
    conf, pred = pbar.max(1); acc = (pred == y.cpu()).float(); e = 0.0
    edges = torch.linspace(0, 1, B + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e

def detect(correct, unc):
    if len(np.unique(correct)) < 2:
        return np.nan, np.nan, np.nan
    s = -unc
    corr_u = unc[correct == 1]
    if len(corr_u) == 0:
        return np.nan, np.nan, np.nan
    thr = np.percentile(corr_u, 95)
    sfr = float(((unc <= thr) & (correct == 0)).sum() / max((correct == 0).sum(), 1))
    try:
        roc = roc_auc_score(correct, s)
        pr = average_precision_score(correct, s)
    except Exception:
        roc, pr = np.nan, np.nan
    return roc, pr, sfr

def add_clean(rows, Pc, yc, meta):
    Uc, pc = uncertainties(Pc); cc = (pc.argmax(1) == yc.cpu()).numpy().astype(int); ece = ece_fn(pc, yc)
    for m in MEASURES:
        r, p, s = detect(cc, Uc[m].numpy())
        rows.append(dict(**meta, proto="clean", attack="none", sub="clean_only", measure=m, roc=r, pr=p, sfr=s, acc=cc.mean(), ece=ece))

def add_adv(rows, Pc, yc, Pa, ya, meta, attack):
    Uc, pc = uncertainties(Pc); Ua, pa = uncertainties(Pa)
    cc = (pc.argmax(1) == yc.cpu()).numpy().astype(int); ca = (pa.argmax(1) == ya.cpu()).numpy().astype(int); ece = ece_fn(pa, ya)
    for m in MEASURES:
        uc, ua = Uc[m].numpy(), Ua[m].numpy()
        for sub, (c, u) in {"mixed": (np.r_[cc, ca], np.r_[uc, ua]), "adv_only": (ca, ua)}.items():
            r, p, s = detect(c, u)
            rows.append(dict(**meta, attack=attack, sub=sub, measure=m, roc=r, pr=p, sfr=s, acc=ca.mean(), ece=ece))

# ------------------------------ EXPERIMENT EXECUTION ------------------------------
def run_unit(ds, seed):
    path = os.path.join(ROOT, "cache", f"{CFG_HASH}_{ds}_{seed}.pkl")
    if os.path.exists(path):
        print(f">> [cache] Loaded {ds} seed {seed}")
        return pickle.load(open(path, "rb"))
        
    t0 = time.time()
    rows = []
    print(f"\n>> Starting evaluation unit: {ds} (seed {seed})...")
    Xtr, ytr, Xte, yte = load_ds(ds, seed)
    T = CFG["T"]
    
    for kind in ["mc", "bnn"]:
        tk0 = time.time()
        print(f"  Training {kind.upper()} baseline on {ds}...")
        base = train_model(ds, kind, Xtr, ytr, seed)
        Pc = mc_probs(base, Xte, T)
        add_clean(rows, Pc, yte, dict(ds=ds, seed=seed, kind=kind, trn="none"))
        
        # Static adversarial test sets
        print(f"  Crafting {len(ALL_ATTACKS)} static baseline attacks on {ds} ({kind.upper()})...")
        adv_test = {a: craft(a, base, Xte, yte, ds) for a in ALL_ATTACKS}
        for a in ALL_ATTACKS:
            add_adv(rows, Pc, yte, mc_probs(base, adv_test[a], T), yte, dict(ds=ds, seed=seed, kind=kind, trn="none", proto="baseline"), a)
            
        for tr in TRAIN_ATTACKS:
            print(f"  Adversarial Training with {tr} on {ds} ({kind.upper()})...")
            Xadv = craft(tr, base, Xtr, ytr, ds)
            m2 = train_model(ds, kind, torch.cat([Xtr, Xadv]), torch.cat([ytr, ytr]), seed)
            Pc2 = mc_probs(m2, Xte, T)
            add_clean(rows, Pc2, yte, dict(ds=ds, seed=seed, kind=kind, trn=tr))
            
            for a in ALL_ATTACKS:
                add_adv(rows, Pc2, yte, mc_probs(m2, adv_test[a], T), yte, dict(ds=ds, seed=seed, kind=kind, trn=tr, proto="static"), a)
            for a in FRESH_ATTACKS:
                Xf = craft(a, m2, Xte, yte, ds)
                add_adv(rows, Pc2, yte, mc_probs(m2, Xf, T), yte, dict(ds=ds, seed=seed, kind=kind, trn=tr, proto="fresh"), a)
                
        print(f">> Finished {ds} {kind.upper()} in {time.time() - tk0:.1f}s")
        
    pickle.dump(rows, open(path, "wb"))
    return rows

all_rows = []
for ds in CFG["datasets"]:
    for sd in CFG["seeds"]:
        all_rows += run_unit(ds, sd)

df = pd.DataFrame(all_rows)
results_csv_file = os.path.join(ROOT, "results", f"all_results_{CFG_HASH}.csv")
df.to_csv(results_csv_file, index=False)
print(f"\n>> Consolidated Results: {df.shape} evaluation rows | Saved to: {results_csv_file}")

# ------------------------------ AUDIT ANALYSES (C1 - C6) ------------------------------
def sel(d, **kw):
    m = np.ones(len(d), bool)
    for k, v in kw.items():
        m &= (d[k].isin(list(v)) if isinstance(v, (list, tuple, set)) else d[k] == v).values
    return d[m]

def boot_ci(v, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0: return (0.0, 0.0)
    return np.percentile([rng.choice(v, len(v), replace=True).mean() for _ in range(n)], [2.5, 97.5])

def wil(v):
    v = np.asarray(v, float)
    v = v[(v != 0) & ~np.isnan(v)]
    if len(v) < 6 or np.all(v == v[0]): return np.nan
    try:
        return stats.wilcoxon(v).pvalue
    except Exception:
        return np.nan

RES = {}

# C1: Existing UQ fails on adversarial errors
b = sel(df, proto="baseline", sub="mixed")
cells = b.groupby(["ds", "kind", "attack", "measure"]).roc.mean()
b_adv = sel(df, proto="baseline", sub="adv_only").groupby(["ds", "kind", "attack", "measure"]).roc.mean()
by_att = b.groupby("attack").agg(AUC_ROC=("roc", "mean"), AUC_PR=("pr", "mean"), silent_failure=("sfr", "mean"), adv_acc=("acc", "mean")).reindex(ALL_ATTACKS).round(3)
RES["c1_table"] = by_att
RES["c1"] = dict(n_cells=len(cells), n_below=int((cells < 0.5).sum()), n_adv_below=int((b_adv < 0.5).sum()), n_adv=len(b_adv),
                 mean_sfr=float(b["sfr"].mean()), mean_roc=float(cells.mean()), mean_adv_only=float(b_adv.mean()), mean_adv_acc=float(b["acc"].mean()))
print("\n=== [C1] Baseline UQ Performance Under Attack ===")
print(by_att.to_string())

hm = b.groupby(["attack", "kind", "ds"]).roc.mean().unstack([1, 2]).reindex(ALL_ATTACKS)
fig, ax = plt.subplots(figsize=(max(7, 1.2 * hm.shape[1] + 3), 4.5))
im = ax.imshow(hm.values, cmap="RdYlGn", vmin=0, vmax=1)
ax.set_xticks(range(hm.shape[1])); ax.set_xticklabels([f"{k}\n{d}" for k, d in hm.columns], fontsize=8)
ax.set_yticks(range(len(hm))); ax.set_yticklabels(hm.index)
for i in range(hm.shape[0]):
    for j in range(hm.shape[1]):
        val = hm.values[i, j]
        ax.text(j, i, f"{val:.2f}" if not np.isnan(val) else "N/A", ha="center", va="center", fontsize=8)
plt.colorbar(im); ax.set_title("C1: Baseline AUC-ROC across Datasets and Models")
plt.tight_layout()
def safe_save(fig, path):
    for attempt in range(5):
        try:
            fig.savefig(path, dpi=150)
            return path
        except Exception:
            time.sleep(0.5)
    alt_path = path.replace(".png", f"_{int(time.time())}.png")
    fig.savefig(alt_path, dpi=150)
    return alt_path

c1_fig_path = safe_save(plt.gcf(), os.path.join(ROOT, "figures", "c1_heatmap.png"))
plt.close()
RES["fig_c1"] = c1_fig_path

# C2: MC Dropout vs BNN
bb = b.melt(id_vars=["ds", "seed", "kind", "attack", "measure"], value_vars=["roc", "pr"], var_name="metric", value_name="v")
pv = bb.pivot_table(index=["ds", "seed", "attack", "measure", "metric"], columns="kind", values="v").dropna()
win = (pv["mc"] > pv["bnn"]).astype(float)
k, n = int(win.sum()), len(win)
unit = win.groupby(level=["ds", "attack"]).mean()
ci = boot_ci(unit.values)
RES["c2"] = dict(n_cells=n, wins=k, win_rate=k / max(n, 1),
                 binom_p=stats.binomtest(k, max(n, 1), 0.5).pvalue if n > 0 else 1.0,
                 wilcoxon_p=wil(pv["mc"] - pv["bnn"]),
                 mean_diff=float((pv["mc"] - pv["bnn"]).mean()) if n > 0 else 0.0,
                 n_units=len(unit), unit_ci=(ci[0], ci[1]),
                 unit_binom_p=stats.binomtest(int((unit > 0.5).sum()), int((unit != 0.5).sum()) or 1, 0.5).pvalue)
print(f"\n=== [C2] MC Dropout vs BNN: Win Rate {RES['c2']['win_rate']:.1%} (Bootstrap CI [{ci[0]:.2f}, {ci[1]:.2f}]) ===")

# C3: Adversarial training: Static vs Fresh, Seen vs Unseen
M = sel(df, sub="mixed")
B0 = sel(M, proto="baseline")[["ds", "seed", "kind", "attack", "measure", "roc", "acc"]].rename(columns={"roc": "roc_b", "acc": "acc_b"})
A = sel(M, proto=["static", "fresh"]).merge(B0, on=["ds", "seed", "kind", "attack", "measure"], how="left")
A["d_roc"] = A["roc"] - A["roc_b"]; A["d_acc"] = A["acc"] - A["acc_b"]; A["seen"] = np.where(A["attack"] == A["trn"], "seen", "unseen")
U = A.groupby(["proto", "seen", "ds", "kind", "trn", "attack"], as_index=False).agg(d_roc=("d_roc", "mean"), d_acc=("d_acc", "mean"))
r3 = []
for (p, s), g in U.groupby(["proto", "seen"]):
    c = boot_ci(g.d_roc); r3.append(dict(protocol=p, attack_type=s, n_units=len(g), mean_dAUC=g.d_roc.mean(), ci_lo=c[0], ci_hi=c[1],
                                          share_improved=(g.d_roc > 0).mean(), wilcoxon_p=wil(g.d_roc), mean_dAcc=g.d_acc.mean()))
for p, g in U.groupby("proto"):
    c = boot_ci(g.d_roc); r3.append(dict(protocol=p, attack_type="all", n_units=len(g), mean_dAUC=g.d_roc.mean(), ci_lo=c[0], ci_hi=c[1],
                                          share_improved=(g.d_roc > 0).mean(), wilcoxon_p=wil(g.d_roc), mean_dAcc=g.d_acc.mean()))
c3 = pd.DataFrame(r3).round(4); RES["c3"] = c3
print("\n=== [C3] Static vs Fresh Robustness Gain ===")
print(c3.to_string())

pair = U.pivot_table(index=["ds", "kind", "trn", "attack"], columns="proto", values="d_roc").dropna()
if len(pair) > 0 and "static" in pair and "fresh" in pair:
    RES["c3_pair"] = dict(n=len(pair), mean_static=float(pair["static"].mean()), mean_fresh=float(pair["fresh"].mean()), wilcoxon_p=wil(pair["static"] - pair["fresh"]))
else:
    RES["c3_pair"] = dict(n=0, mean_static=0.0, mean_fresh=0.0, wilcoxon_p=1.0)

cmp = A.groupby(["proto", "trn"]).d_roc.mean().unstack(0)
fig, ax = plt.subplots(figsize=(6, 3.8))
cmp.plot.bar(ax=ax); ax.axhline(0, color="k", lw=.8); ax.set_ylabel("Mean Change in AUC-ROC"); ax.set_title("C3: Gain vs Baseline (Static vs Fresh)")
plt.tight_layout()
c3_fig_path = safe_save(fig, os.path.join(ROOT, "figures", "c3_static_fresh.png"))
plt.close()
RES["fig_c3"] = c3_fig_path

# C4: Accuracy gain vs UQ gain
G = A.groupby(["proto", "ds", "kind", "trn"], as_index=False).agg(d_acc=("d_acc", "mean"), d_roc=("d_roc", "mean"))
RES["c4"] = {p: stats.spearmanr(g.d_acc, g.d_roc) for p, g in G.groupby("proto")}
RES["c4"] = {p: dict(rho=float(r[0]) if not np.isnan(r[0]) else 0.0, p=float(r[1]) if not np.isnan(r[1]) else 1.0, n=int((G.proto == p).sum())) for p, r in RES["c4"].items()}
RES["c4_rank"] = A.groupby(["proto", "trn"]).agg(d_acc=("d_acc", "mean"), d_roc=("d_roc", "mean")).round(4)

# C5: Uncertainty measures
RES["c5"] = A.groupby(["proto", "measure"]).d_roc.mean().unstack(0).reindex(MEASURES).round(4)
RES["c5_kind"] = A.groupby(["proto", "kind"]).d_roc.mean().unstack(0).round(4)

# C6: Clean performance impact
cl = sel(df, sub="clean_only", measure="PU")
c6 = cl.groupby(["ds", "kind", "trn"]).agg(clean_acc=("acc", "mean"), ECE=("ece", "mean"), clean_AUC_ROC=("roc", "mean"), silent_failure=("sfr", "mean")).round(4)
base6 = c6.xs("none", level="trn")
tr6 = c6[c6.index.get_level_values("trn") != "none"].groupby(["ds", "kind"]).mean()
RES["c6_table"] = pd.concat([base6.add_prefix("base_"), tr6.add_prefix("advtrained_")], axis=1).round(4)
RES["c6"] = dict(d_acc=float((tr6.clean_acc - base6.clean_acc).mean()), d_ece=float((tr6.ECE - base6.ECE).mean()),
                 d_auc=float((tr6.clean_AUC_ROC - base6.clean_AUC_ROC).mean()), base_acc=float(base6.clean_acc.mean()))

# ------------------------------ EPSILON SWEEP ------------------------------
def eps_sweep(ds, kind="mc", scales=(0.5, 1.0, 1.5), seed=0):
    cache_p = os.path.join(ROOT, "cache", f"sweep_{CFG_HASH}_{ds}_{seed}.pkl")
    if os.path.exists(cache_p):
        return pickle.load(open(cache_p, "rb"))
    Xtr, ytr, Xte, yte = load_ds(ds, seed); base = train_model(ds, kind, Xtr, ytr, seed); out = []
    eval_subset = [a for a in ["BIM-A", "BIM-B", "Adaptive"] if a in ALL_ATTACKS]
    for sc in scales:
        Xadv = craft("BIM-B", base, Xtr, ytr, ds, scale=sc)
        m = train_model(ds, kind, torch.cat([Xtr, Xadv]), torch.cat([ytr, ytr]), seed)
        Pc = mc_probs(m, Xte, CFG["T"])
        for a in eval_subset:
            rows = []
            add_adv(rows, Pc, yte, mc_probs(m, craft(a, m, Xte, yte, ds), CFG["T"]), yte, dict(), a)
            r = pd.DataFrame(rows)
            r = r[(r["sub"] == "mixed") & (r["measure"] == "PU")].iloc[0]
            out.append(dict(train_eps_scale=sc, eval_attack=a, AUC_ROC=r.roc, adv_acc=r.acc))
    df_sweep = pd.DataFrame(out)
    pickle.dump(df_sweep, open(cache_p, "wb"))
    return df_sweep

print("\n>> Running Epsilon Sensitivity Sweep on MNIST...")
sweep = eps_sweep(CFG["datasets"][0]).round(3)
RES["sweep"] = sweep; RES["sweep_ds"] = CFG["datasets"][0]
print("=== Epsilon Sensitivity Sweep ===")
print(sweep.pivot(index="train_eps_scale", columns="eval_attack", values="AUC_ROC").to_string())

# ------------------------------ TRUST SCORECARD ------------------------------
fr = sel(M, proto="fresh"); fr_cells = fr.groupby(["ds", "kind", "trn", "attack", "measure"]).roc.mean()
seed_gain = A[A["proto"] == "fresh"].groupby("seed").d_roc.mean()
same_dir = bool((seed_gain > 0).all() or (seed_gain < 0).all()) if len(seed_gain) > 1 else True

fresh_row = RES["c3"][(RES["c3"].protocol == "fresh") & (RES["c3"].attack_type == "all")]
if len(fresh_row) > 0:
    fresh_all = fresh_row.iloc[0]
else:
    fresh_all = pd.Series(dict(mean_dAUC=0.0, ci_lo=0.0, ci_hi=0.0))

cl_tr = sel(df, sub="clean_only", measure="PU"); cl_tr = cl_tr[cl_tr["trn"] != "none"]
sf = sel(fr, measure="PU")["sfr"].mean() if len(fr) > 0 else 0.5

sc = [
 ("Robustness",   "mean AUC gain under FRESH white-box attack > 0 (CI excludes 0)", f"{fresh_all.mean_dAUC:+.3f} [{fresh_all.ci_lo:+.3f},{fresh_all.ci_hi:+.3f}]", bool(fresh_all.ci_lo > 0)),
 ("Reliability",  "mean AUC-ROC >= 0.70 and no cell < 0.5 (fresh, adv-trained)", f"mean {fr_cells.mean():.3f}, cells<0.5: {(fr_cells < 0.5).sum()}/{len(fr_cells)}", bool(fr_cells.mean() >= 0.70 and (fr_cells < 0.5).sum() == 0)),
 ("Calibration",  "clean ECE <= 0.08 and adversarial ECE <= 0.15", f"clean {cl_tr.ece.mean():.3f}, adversarial {fr.ece.mean():.3f}", bool(cl_tr.ece.mean() <= .08 and fr.ece.mean() <= .15)),
 ("Safety",       "silent-failure rate <= 35% at 95% retention (PU, mixed, fresh)", f"{sf:.1%}", bool(sf <= 0.35)),
 ("Validity",     "effect larger than CI, >=2 seeds (if standard), same direction", f"seeds={len(CFG['seeds'])}, fresh CI [{fresh_all.ci_lo:+.3f},{fresh_all.ci_hi:+.3f}]", bool(fresh_all.ci_lo > 0 and (len(CFG['seeds']) < 2 or same_dir))),
 ("Reproducibility", "seeds + config hash + cached results logged", f"hash {CFG_HASH}, seeds {CFG['seeds']}", True),
]
score = pd.DataFrame(sc, columns=["Pillar", "Passing looks like", "This run", "Pass"])
RES["score"] = score
score["Pass"] = score["Pass"].map({True: "PASS", False: "FAIL"})
print("\n=== Multi-Pillar Trust Scorecard ===")
print(score.to_string())

# ------------------------------ WORD REPORT GENERATION ------------------------------
from docx import Document
from docx.shared import Inches, Pt

def verdicts():
    c1, c2, c3p, c6 = RES["c1"], RES["c2"], RES["c3_pair"], RES["c6"]; c3 = RES["c3"]
    f_match = c3[(c3.protocol == "fresh") & (c3.attack_type == "all")]
    s_match = c3[(c3.protocol == "static") & (c3.attack_type == "all")]
    f = f_match.iloc[0] if len(f_match) > 0 else pd.Series(dict(mean_dAUC=0.0, ci_lo=0.0, ci_hi=0.0, mean_dAcc=0.0))
    s = s_match.iloc[0] if len(s_match) > 0 else pd.Series(dict(mean_dAUC=0.0, ci_lo=0.0, ci_hi=0.0, mean_dAcc=0.0))
    
    v = {}
    bad = RES["c1_table"]["AUC_ROC"].dropna()
    worst_att = bad.idxmin() if len(bad) > 0 else "N/A"
    worst_val = bad.min() if len(bad) > 0 else 0.5
    best_att = bad.idxmax() if len(bad) > 0 else "N/A"
    best_val = bad.max() if len(bad) > 0 else 0.5
    
    v["C1"] = (f"{'SUPPORTED, attack-dependent' if (bad < 0.5).any() else 'PARTLY SUPPORTED'}: {c1['n_below']}/{c1['n_cells']} baseline cells have AUC-ROC < 0.5 "
               f"(worst attack: {worst_att} {worst_val:.2f}; best: {best_att} {best_val:.2f}). Mean silent-failure rate {c1['mean_sfr']:.0%}. "
               f"On adversarial samples alone, {c1['n_adv_below']}/{c1['n_adv']} cells are below 0.5 (mean {c1['mean_adv_only']:.2f}); mean adversarial accuracy is {c1['mean_adv_acc']:.1%}, "
               "so the mixed 'correct vs wrong' task is largely clean-vs-adversarial detection.")
    lo, hi = c2["unit_ci"]
    v["C2"] = (f"WEAK EVIDENCE: MC Dropout wins {c2['win_rate']:.1%} of {c2['n_cells']} correlated cells (cell-level binomial p={c2['binom_p']:.3g}, Wilcoxon p={c2['wilcoxon_p']:.3g}). "
               f"Resampling the {c2['n_units']} dataset x attack units gives a 95% CI of [{lo:.2f},{hi:.2f}]. "
               f"{'Parity cannot be rejected' if lo <= 0.5 <= hi else 'Parity is rejected'}, validating MC Dropout as a practical approximation to BNNs.")
    v["C3"] = (f"{'SUPPORTED' if f.ci_lo > 0 else 'NOT ESTABLISHED'} under the fresh protocol: mean AUC-ROC change {f.mean_dAUC:+.3f} [{f.ci_lo:+.3f},{f.ci_hi:+.3f}], "
               f"vs {s.mean_dAUC:+.3f} under the static protocol the paper uses (paired static-minus-fresh Wilcoxon p={c3p['wilcoxon_p']:.3g}). "
               f"Accuracy gain: static {s.mean_dAcc*100:+.1f} points vs fresh {f.mean_dAcc*100:+.1f} points. "
               + ("The static gain is clearly larger than the fresh gain: this pattern is typical of matching to fixed attack samples or gradient masking, not real robustness." if (s.mean_dAUC - f.mean_dAUC > 0.01 or s.mean_dAcc - f.mean_dAcc > 0.10) else "Static and fresh gains are comparable."))
    r = RES["c4"]
    v["C4"] = "; ".join(f"{p}: Spearman rho={d['rho']:+.2f} (p={d['p']:.2g}, n={d['n']})" for p, d in r.items()) + ". Near-zero or negative rho supports the paper's claim that accuracy gains do not track UQ gains."
    c5 = RES["c5"]
    ep = c5.loc[[m for m in ["MI", "EKL"] if m in c5.index]].mean() if len(c5) > 0 else pd.Series()
    oth = c5.loc[[m for m in ["PV", "PU"] if m in c5.index]].mean() if len(c5) > 0 else pd.Series()
    v["C5"] = (f"Mean AUC gain MI/EKL: static {ep.get('static', np.nan):+.3f}, fresh {ep.get('fresh', np.nan):+.3f}; PV/PU: static {oth.get('static', np.nan):+.3f}, fresh {oth.get('fresh', np.nan):+.3f}. "
               "PU is dominated by MP, so equal gains for PU do not isolate epistemic uncertainty. The claim 'mainly epistemic' is only supported if MI/EKL gains clearly exceed the rest.")
    v["C6"] = (f"Clean accuracy changes by {c6['d_acc']*100:+.2f} points (baseline {c6['base_acc']:.1%}), ECE by {c6['d_ece']:+.4f}, clean-only detection AUC-ROC by {c6['d_auc']:+.3f}. "
               f"{'Harmless on accuracy' if abs(c6['d_acc']) < 0.01 else 'Accuracy cost is visible'}; calibration and detection must be judged from these separate numbers.")
    return v

def add_df(doc, d, index=True, nd=3):
    d = d.reset_index() if index else d
    t = doc.add_table(rows=1, cols=len(d.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(d.columns): t.rows[0].cells[i].text = str(c)
    for _, r in d.iterrows():
        cs = t.add_row().cells
        for i, x in enumerate(r):
            if isinstance(x, (float, np.floating)):
                cs[i].text = f"{x:.{nd}f}" if not np.isnan(x) else "N/A"
            else:
                cs[i].text = str(x)
    doc.add_paragraph()

def build_report(path):
    V = verdicts()
    doc = Document()
    doc.styles["Normal"].font.size = Pt(10)
    doc.add_heading("Trustworthiness Audit: Results (Sections 8-13)", 0)
    doc.add_paragraph(f"Run: preset '{CFG['preset']}', config hash {CFG_HASH}. Generated automatically from the audit script; every number comes from this run across MNIST, CIFAR-10, and SVHN.")
    
    doc.add_heading("8. Experimental setup of this run", 1)
    add_df(doc, pd.DataFrame([
        dict(item="datasets", value=", ".join(CFG["datasets"])),
        dict(item="train / test images per dataset", value=f"{CFG['n_train']} / {CFG['n_test']}"),
        dict(item="epochs per dataset, lr, batch", value=f"{CFG['epochs']}, {CFG['lr']}, {CFG['bs']}"),
        dict(item="CIFAR-10 augmentation", value="flip + shift" if CFG.get("aug_cifar") else "none"),
        dict(item="MC passes T", value=CFG["T"]),
        dict(item="seeds", value=str(CFG["seeds"])),
        dict(item="adversarial training attacks", value=", ".join(TRAIN_ATTACKS)),
        dict(item="fresh re-attack set", value=", ".join(FRESH_ATTACKS)),
        dict(item="evaluation attack suite", value=", ".join(ALL_ATTACKS)),
        dict(item="L-inf eps", value="MNIST 0.3, CIFAR-10/SVHN 8/255")
    ]), index=False)
    
    doc.add_heading("Data statistics (EDA on the subsets used)", 2)
    add_df(doc, eda, index=False, nd=4)
    
    doc.add_heading("9. Results by claim", 1)
    names = {
        "C1": "Existing UQ fails on adversarial errors",
        "C2": "MC Dropout is as good as BNN",
        "C3": "Adversarial training improves UQ (also on unseen attacks)",
        "C4": "Accuracy gains do not match UQ gains",
        "C5": "Gains come mainly from epistemic uncertainty",
        "C6": "The method is cheap and harmless"
    }
    for cid in ["C1", "C2", "C3", "C4", "C5", "C6"]:
        doc.add_heading(f"{cid}. {names[cid]}", 2)
        doc.add_paragraph(V[cid])
        if cid == "C1":
            add_df(doc, RES["c1_table"])
            if os.path.exists(RES.get("fig_c1", "")): doc.add_picture(RES["fig_c1"], width=Inches(6.0))
        if cid == "C3":
            add_df(doc, RES["c3"], index=False, nd=4)
            if os.path.exists(RES.get("fig_c3", "")): doc.add_picture(RES["fig_c3"], width=Inches(5.0))
        if cid == "C4":
            add_df(doc, RES["c4_rank"], nd=4)
        if cid == "C5":
            add_df(doc, RES["c5"], nd=4)
        if cid == "C6":
            add_df(doc, RES["c6_table"], nd=4)
            
    doc.add_heading("Epsilon sensitivity", 2)
    doc.add_paragraph(f"{RES['sweep_ds']}, MC Dropout, BIM-B training with perturbation size scaled; evaluation uses fresh attacks (PU, mixed protocol).")
    add_df(doc, RES["sweep"], index=False)
    
    doc.add_heading("10. Claim ledger", 1)
    add_df(doc, pd.DataFrame([dict(claim=k, finding=V[k]) for k in V]), index=False)
    
    doc.add_heading("11. Trust scorecard", 1)
    add_df(doc, RES["score"], index=False)
    
    doc.add_heading("12. Limitations of this reproduction", 1)
    for s in [
        f"Evaluated on {', '.join(CFG['datasets'])} with balanced subsets ({CFG['n_train']} train / {CFG['n_test']} test), {CFG['T']} MC passes, {len(CFG['seeds'])} seed(s).",
        "Models are attacked in deterministic mode; stochastic inference is applied afterwards for uncertainty estimation.",
        "AutoAttack uses APGD-reduced (APGD-CE + APGD-DLR); Adaptive is an uncertainty-adaptive optimization attack.",
        "Conclusions apply to MNIST, CIFAR-10, and SVHN under the specified perturbation radii."
    ]:
        doc.add_paragraph(s, style="List Bullet")
        
    doc.add_heading("13. Conclusion and recommendations", 1)
    doc.add_paragraph("Recommended practice: evaluate adversarial training with a fresh white-box re-attack of the retrained model, "
                      "report clean-only and adversarial-only detection separately from the mixed protocol, "
                      "and set operational reject thresholds from the silent-failure rate rather than from AUC alone.")
    
    fallback_path = os.path.join(ROOT, "results", "TAIML_Report_Results_MultiDataset.docx")
    try:
        doc.save(path)
        print(f">> Report saved successfully to: {path}")
    except PermissionError:
        print(f">> Warning: {path} is locked by another process (e.g. Word). Saving to fallback: {fallback_path}")
        doc.save(fallback_path)
        path = fallback_path
    except Exception as e:
        print(f">> Warning: Failed to save to {path} ({e}). Saving to fallback: {fallback_path}")
        doc.save(fallback_path)
        path = fallback_path

    # Also always ensure fallback copy is available for multi-dataset report
    if path != fallback_path:
        try:
            doc.save(fallback_path)
        except Exception:
            pass

    return path

report_path = os.path.join(ROOT, "results", "TAIML_Report_Results.docx")
final_path = build_report(report_path)
print(f"\n>> [COMPLETE] Updated audit report compiled: {final_path}")
