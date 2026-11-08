import re
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm

from data import MEAN, STD
from model import get_model

EPOCHS = 15


# ---------- data ----------
def kaggle_splits():
    """Same 70/15/15 split as data.py (seed 42)."""
    full = datasets.ImageFolder("data/raw")
    idx = np.random.RandomState(42).permutation(len(full))
    n_train, n_val = int(0.7 * len(idx)), int(0.15 * len(idx))
    parts = [idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]]
    return [[full.samples[i] for i in p] for p in parts]


def group_key(path, label):
    """COVID: source paper from filename. non-COVID: figure number (partial)."""
    stem = Path(path).stem
    if label == 0:
        m = re.match(r"(.+)-p\d+-\d+", stem)
        return "C_" + (m.group(1) if m else stem)
    return "N_" + stem.split("%")[0]


def external_splits(seed=42):
    """Group-aware split of the external set: ~50% train / 20% val / 30% test."""
    samples = datasets.ImageFolder("data/external").samples
    groups = [group_key(p, l) for p, l in samples]
    idx = np.arange(len(samples))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    trainval, test = next(gss.split(idx, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, va = next(gss2.split(trainval, groups=[groups[i] for i in trainval]))

    pick = lambda ii: [samples[i] for i in ii]
    return pick(trainval[tr]), pick(trainval[va]), pick(test)


class ListDS(Dataset):
    def __init__(self, samples, tf):
        self.samples, self.tf = samples, tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        return self.tf(Image.open(path).convert("RGB")), y


def eval_tf():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def strong_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), value=0),
    ])


# ---------- evaluation ----------
@torch.no_grad()
def predict(model, samples, device):
    dl = DataLoader(ListDS(samples, eval_tf()), batch_size=32,
                    shuffle=False, num_workers=0)
    model.eval()
    probs, ys = [], []
    for x, y in dl:
        probs += torch.softmax(model(x.to(device)), dim=1)[:, 0].cpu().tolist()
        ys += y.tolist()
    return np.array(ys), np.array(probs)


def report(tag, y, p):
    covid = (y == 0).astype(int)
    pred = (p > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(covid, pred, labels=[0, 1]).ravel()
    res = dict(n=len(y), acc=(tp + tn) / len(y), sens=tp / (tp + fn),
               spec=tn / (tn + fp), auc=roc_auc_score(covid, p))
    print(f"{tag:14s} n={res['n']:4d} | acc {res['acc']:.3f} | "
          f"sens {res['sens']:.3f} | spec {res['spec']:.3f} | AUC {res['auc']:.3f}")
    return res


# ---------- training ----------
def train(model, train_samples, val_samples, ckpt, device):
    train_dl = DataLoader(ListDS(train_samples, strong_tf()), batch_size=32,
                          shuffle=True, num_workers=0)
    val_dl = DataLoader(ListDS(val_samples, eval_tf()), batch_size=32,
                        shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        tr_loss, tr_correct = 0.0, 0
        for x, y in tqdm(train_dl, leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * len(y)
            tr_correct += (out.argmax(1) == y).sum().item()

        model.eval()
        va_loss, va_correct = 0.0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                out = model(x)
                va_loss += criterion(out, y).item() * len(y)
                va_correct += (out.argmax(1) == y).sum().item()
        va_loss /= len(val_samples)
        scheduler.step(va_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | train loss {tr_loss/len(train_samples):.3f} "
              f"acc {tr_correct/len(train_samples):.3f} | val loss {va_loss:.3f} "
              f"acc {va_correct/len(val_samples):.3f}")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), ckpt)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    assert mode in {"baseline", "aug", "mixed"}, "mode: baseline | aug | mixed"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    k_tr, k_va, k_te = kaggle_splits()
    e_tr, e_va, e_te = external_splits()
    print(f"Mode: {mode} | device: {device}")
    print(f"Kaggle train/val/test: {len(k_tr)}/{len(k_va)}/{len(k_te)} | "
          f"External train/val/test: {len(e_tr)}/{len(e_va)}/{len(e_te)}")

    model = get_model("resnet").to(device)
    if mode == "baseline":
        ckpt = "outputs/resnet_best.pt"
    else:
        ckpt = f"outputs/resnet_{mode}_best.pt"
        train_s = k_tr + (e_tr if mode == "mixed" else [])
        val_s = k_va + (e_va if mode == "mixed" else [])
        train(model, train_s, val_s, ckpt, device)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f"\n=== Results for '{mode}' ===")
    results = {
        "kaggle_test": report("Kaggle test", *predict(model, k_te, device)),
        "external_test": report("External test", *predict(model, e_te, device)),
    }
    with open(f"outputs/results_{mode}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()