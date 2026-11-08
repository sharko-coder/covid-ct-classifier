import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (confusion_matrix, roc_auc_score, roc_curve,
                             ConfusionMatrixDisplay)

from data import MEAN, STD
from model import get_model

EXT_ROOT = "data/external"


def evaluate(name, loader, device):
    model = get_model(name).to(device)
    model.load_state_dict(torch.load(f"outputs/{name}_best.pt", map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            p = torch.softmax(model(x.to(device)), dim=1)[:, 0].cpu()  # P(COVID)
            y_prob += p.tolist()
            y_true += y.tolist()
    return np.array(y_true), np.array(y_prob)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder(EXT_ROOT, transform=tf)
    print("Classes:", ds.classes, "| images:", len(ds))
    assert ds.classes == ["COVID", "non-COVID"], "Folder names must be COVID and non-COVID"
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for row, name in enumerate(["cnn", "resnet"]):
        y_true, y_prob = evaluate(name, loader, device)
        covid_true = (y_true == 0).astype(int)
        covid_pred = (y_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(covid_true, covid_pred).ravel()

        print(f"\n=== {name} on EXTERNAL dataset ({len(y_true)} images) ===")
        print(f"Accuracy:    {(tp + tn) / len(y_true):.3f}")
        print(f"Sensitivity: {tp / (tp + fn):.3f}")
        print(f"Specificity: {tn / (tn + fp):.3f}")
        print(f"ROC-AUC:     {roc_auc_score(covid_true, y_prob):.3f}")
        print(f"Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")

        ConfusionMatrixDisplay(confusion_matrix(covid_true, covid_pred),
                               display_labels=["non-COVID", "COVID"]).plot(ax=axes[row, 0])
        axes[row, 0].set_title(f"{name}: external confusion matrix")
        fpr, tpr, _ = roc_curve(covid_true, y_prob)
        axes[row, 1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(covid_true, y_prob):.3f}")
        axes[row, 1].plot([0, 1], [0, 1], "--", color="gray")
        axes[row, 1].set(xlabel="False positive rate", ylabel="True positive rate",
                         title=f"{name}: external ROC")
        axes[row, 1].legend()

    plt.tight_layout()
    plt.savefig("outputs/cross_dataset_eval.png", dpi=150, bbox_inches="tight")
    print("\nSaved outputs/cross_dataset_eval.png")


if __name__ == "__main__":
    main()