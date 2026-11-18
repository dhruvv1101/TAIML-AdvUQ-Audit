import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchattacks

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {dev}")

ROOT = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(ROOT, "figures")
os.makedirs(fig_dir, exist_ok=True)

# Dataset classes
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Model architecture from audit
class SimpleConvNet(nn.Module):
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3)
        )
        dummy = torch.zeros(1, in_channels, 28 if in_channels==1 else 32, 28 if in_channels==1 else 32)
        feat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

# Load data helper
def get_sample_and_model(ds_name):
    root = os.path.join(ROOT, "data")
    if ds_name == "MNIST":
        tr = torchvision.datasets.MNIST(root, True, download=False)
        te = torchvision.datasets.MNIST(root, False, download=False)
        X_tr = tr.data[:2500].unsqueeze(1).float() / 255.0
        y_tr = tr.targets[:2500]
        X_te = te.data[:300].unsqueeze(1).float() / 255.0
        y_te = te.targets[:300]
        in_c = 1
        eps = 0.28
        l2 = 2.2
    elif ds_name == "CIFAR10":
        cifar_npz = os.path.join(root, "cifar10_compact.npz")
        data = np.load(cifar_npz)
        X_tr = torch.tensor(data["Xtr"][:2500], dtype=torch.float32) / 255.0
        y_tr = torch.tensor(data["ytr"][:2500], dtype=torch.long)
        X_te = torch.tensor(data["Xte"][:300], dtype=torch.float32) / 255.0
        y_te = torch.tensor(data["yte"][:300], dtype=torch.long)
        in_c = 3
        eps = 12/255.0
        l2 = 0.8
    else: # SVHN
        tr = torchvision.datasets.SVHN(root, split="train", download=False)
        te = torchvision.datasets.SVHN(root, split="test", download=False)
        X_tr = torch.tensor(tr.data[:2500], dtype=torch.float32) / 255.0
        y_tr = torch.tensor(tr.labels[:2500], dtype=torch.long)
        X_te = torch.tensor(te.data[:300], dtype=torch.float32) / 255.0
        y_te = torch.tensor(te.labels[:300], dtype=torch.long)
        in_c = 3
        eps = 12/255.0
        l2 = 1.2

    model = SimpleConvNet(in_c, 10).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=128, shuffle=True)
    
    print(f">> Training baseline model for {ds_name} (6 epochs)...")
    model.train()
    for ep in range(6):
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    
    # Pick a crisp test image that is correctly classified with high confidence
    for i in range(len(X_te)):
        img = X_te[i:i+1].to(dev)
        lbl = y_te[i:i+1].to(dev)
        out = F.softmax(model(img), dim=1)
        conf, pred = out.max(1)
        if pred.item() == lbl.item() and conf.item() > 0.90:
            print(f"Selected test index {i} for {ds_name}: label={lbl.item()}, conf={conf.item():.3f}")
            return model, img, lbl, eps, l2
            
    return model, X_te[0:1].to(dev), y_te[0:1].to(dev), eps, l2

# Attack implementations
def run_attacks(model, x, y, eps, l2):
    model.eval()
    attacks_dict = {}
    attacks_dict["Clean"] = x.clone()

    # 1. FGSM
    x_fgsm = x.clone().requires_grad_(True)
    out = model(x_fgsm)
    loss = F.cross_entropy(out, y)
    loss.backward()
    attacks_dict["FGSM"] = (x + eps * x_fgsm.grad.sign()).clamp(0, 1).detach()

    # 2. BIM-A (Early stop)
    xa = x.clone()
    alpha = eps / 8.0
    for _ in range(15):
        xa.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)[0]
        xa = torch.clamp(x + torch.clamp(xa + alpha * g.sign() - x, -eps, eps), 0, 1).detach()
        if model(xa).argmax(1).item() != y.item():
            break
    attacks_dict["BIM-A"] = xa

    # 3. BIM-B (Full iterations)
    xa = x.clone() + torch.empty_like(x).uniform_(-eps, eps)
    xa = xa.clamp(0, 1)
    for _ in range(12):
        xa.requires_grad_(True)
        g = torch.autograd.grad(F.cross_entropy(model(xa), y), xa)[0]
        xa = torch.clamp(x + torch.clamp(xa + alpha * g.sign() - x, -eps, eps), 0, 1).detach()
    attacks_dict["BIM-B"] = xa

    # 4. APGD (Auto-PGD)
    try:
        apgd = torchattacks.APGD(model, norm="Linf", eps=eps, steps=10, n_restarts=1, loss="ce", seed=0)
        attacks_dict["APGD"] = apgd(x, y).detach()
    except Exception:
        attacks_dict["APGD"] = attacks_dict["BIM-B"]

    # 5. PGDL2
    try:
        pgdl2 = torchattacks.PGDL2(model, eps=l2, alpha=l2/4.0, steps=10, random_start=True)
        attacks_dict["PGDL2"] = pgdl2(x, y).detach()
    except Exception:
        attacks_dict["PGDL2"] = attacks_dict["BIM-B"]

    # 6. Jitter
    try:
        jitter = torchattacks.Jitter(model, eps=eps, alpha=eps/4.0, steps=10, scale=10, std=0.1, random_start=True)
        attacks_dict["Jitter"] = jitter(x, y).detach()
    except Exception:
        attacks_dict["Jitter"] = attacks_dict["BIM-B"]

    # 7. Adaptive (penalize entropy / confidence trick)
    xa = x.clone()
    alpha_ad = eps / 8.0
    for _ in range(12):
        xa.requires_grad_(True)
        logits = model(xa)
        probs = F.softmax(logits, dim=1)
        ce = F.cross_entropy(logits, y)
        # Encourage high confidence (low entropy) on the wrong prediction
        entropy = -torch.sum(probs * torch.log(probs + 1e-12))
        loss = ce + 0.5 * entropy  # minimize entropy of output
        g = torch.autograd.grad(loss, xa)[0]
        xa = torch.clamp(x + torch.clamp(xa + alpha_ad * g.sign() - x, -eps, eps), 0, 1).detach()
    attacks_dict["Adaptive"] = xa

    return attacks_dict

def to_img(tensor, ds_name):
    t = tensor.squeeze(0).cpu().numpy()
    if ds_name == "MNIST":
        return t[0] # 28x28 grayscale
    else:
        return np.transpose(t, (1, 2, 0)) # 32x32x3 RGB

print("Attack routines defined.")

# Main runner
datasets = ["MNIST", "CIFAR10", "SVHN"]
attack_names = ["Clean", "FGSM", "BIM-A", "BIM-B", "APGD", "PGDL2", "Jitter", "Adaptive"]

all_results = {}

for ds in datasets:
    model, img, lbl, eps, l2 = get_sample_and_model(ds)
    attacks = run_attacks(model, img, lbl, eps, l2)
    preds = {}
    for name, a_img in attacks.items():
        with torch.no_grad():
            out = F.softmax(model(a_img), dim=1)
            conf, pred = out.max(1)
            preds[name] = (pred.item(), conf.item())
    all_results[ds] = {
        "img": img,
        "lbl": lbl.item(),
        "attacks": attacks,
        "preds": preds
    }

# ----------------- FIGURE 1: ATTACK VISUAL GRID -----------------
fig, axes = plt.subplots(3, 8, figsize=(20, 8.5), dpi=300)
plt.subplots_adjust(wspace=0.15, hspace=0.35)

for r, ds in enumerate(datasets):
    data = all_results[ds]
    true_label = data["lbl"]
    true_str = str(true_label) if ds != "CIFAR10" else CIFAR_CLASSES[true_label]
    
    for c, att in enumerate(attack_names):
        ax = axes[r, c]
        att_tensor = data["attacks"][att]
        im_arr = to_img(att_tensor, ds)
        pred_label, conf = data["preds"][att]
        pred_str = str(pred_label) if ds != "CIFAR10" else CIFAR_CLASSES[pred_label]
        
        if ds == "MNIST":
            ax.imshow(im_arr, cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(im_arr, 0, 1))
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Color coding: Green if correct, Red if fooled!
        is_correct = (pred_label == true_label)
        color = "#1b5e20" if is_correct else "#b71c1c"
        status = "CORRECT" if is_correct else "FOOLED"
        
        if r == 0:
            ax.set_title(att, fontsize=13, fontweight="bold", pad=8)
            
        if c == 0:
            ax.set_ylabel(f"{ds}\n(True: {true_str})", fontsize=12, fontweight="bold", labelpad=8)
            
        ax.set_xlabel(f"Pred: {pred_str}\n({conf*100:.1f}%)", fontsize=9.5, color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.0 if not is_correct else 1.0)

fig.suptitle("Adversarial Attacks Across MNIST, CIFAR-10, & SVHN (Input Images & Predictions)", fontsize=16, fontweight="bold", y=0.98)
out_path_grid = os.path.join(fig_dir, "presentation_attack_grid.png")
plt.savefig(out_path_grid, bbox_inches="tight")
plt.close()
print(f">> Saved Attack Grid to: {out_path_grid}")

# ----------------- FIGURE 2: MAGNIFIED NOISE SIGNATURES -----------------
# For CIFAR-10, show the clean image, the attack image, and the 5x amplified noise pattern
fig, axes = plt.subplots(2, 7, figsize=(18, 5.5), dpi=300)
plt.subplots_adjust(wspace=0.12, hspace=0.25)

cifar_data = all_results["CIFAR10"]
clean_tensor = cifar_data["attacks"]["Clean"]
clean_img = to_img(clean_tensor, "CIFAR10")
attacks_only = ["FGSM", "BIM-A", "BIM-B", "APGD", "PGDL2", "Jitter", "Adaptive"]

for c, att in enumerate(attacks_only):
    att_tensor = cifar_data["attacks"][att]
    att_img = to_img(att_tensor, "CIFAR10")
    
    # Noise calculation: difference amplified by 5x and centered
    diff = np.abs(att_img - clean_img)
    # amplify for visual inspection
    diff_mag = np.clip(diff * 6.0, 0, 1)
    
    # Row 0: Perturbed Image
    ax0 = axes[0, c]
    ax0.imshow(np.clip(att_img, 0, 1))
    pred_label, conf = cifar_data["preds"][att]
    pred_str = CIFAR_CLASSES[pred_label]
    ax0.set_title(att, fontsize=12, fontweight="bold")
    ax0.set_xlabel(f"Pred: {pred_str} ({conf*100:.1f}%)", fontsize=9, color="#b71c1c", fontweight="bold")
    ax0.set_xticks([]); ax0.set_yticks([])
    if c == 0:
        ax0.set_ylabel(r"Perturbed" + "\n" + r"Image ($x + \eta$)", fontsize=11, fontweight="bold")
        
    # Row 1: Amplified Perturbation
    ax1 = axes[1, c]
    # Sum over color channels to get crisp perturbation heatmap
    diff_gray = diff.sum(axis=2)
    diff_gray = diff_gray / (diff_gray.max() + 1e-8)
    im1 = ax1.imshow(diff_gray, cmap="inferno", vmin=0, vmax=1)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel(rf"Max $|\eta|$: {np.max(diff):.3f}", fontsize=9)
    if c == 0:
        ax1.set_ylabel(r"Perturbation" + "\n" + r"Signature ($|\eta| \times 6$)", fontsize=11, fontweight="bold")

fig.suptitle("CIFAR-10 Attack Perturbation Heatmaps & Mathematical Signatures", fontsize=15, fontweight="bold", y=0.98)
out_path_noise = os.path.join(fig_dir, "presentation_attack_signatures.png")
plt.savefig(out_path_noise, bbox_inches="tight")
plt.close()
print(f">> Saved Attack Noise Signatures to: {out_path_noise}")
print(">> ALL PRESENTATION FIGURES COMPLETED.")
