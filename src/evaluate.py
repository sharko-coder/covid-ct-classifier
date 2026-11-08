import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, ConfusionMatrixDisplay)

from data import get_loaders
from model import get_model


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cnn"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    (_, _, test_dl), classes = get_loaders()
    model = get_model(name).to(device)
    model.load_state_dict(torch.load(f"outputs/{name}_best.pt", map_location=device))
    model.eval()

    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in test_dl:
            # class 0 = COVID (alphabetical order), so column 0 = P(COVID)
            p = torch.softmax(model(x.to(device)), dim=1)[:, 0].cpu()
            y_prob += p.tolist()
            y_true += y.tolist()

    y_true, y_prob = np.array(y_true), np.array(y_prob)
    covid_true = (y_true == 0).astype(int)   # 1 = COVID
    covid_pred = (y_prob > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(covid_true, covid_pred).ravel()
    print(f"\n=== {name} on TEST set ({len(y_true)} images) ===")
    print(classification_report(covid_true, covid_pred,
                                target_names=["non-COVID", "COVID"], digits=3))
    print(f"Sensitivity (COVID recall): {tp / (tp + fn):.3f}")
    print(f"Specificity:                {tn / (tn + fp):.3f}")
    print(f"ROC-AUC:                    {roc_auc_score(covid_true, y_prob):.3f}")
    print(f"Confusion matrix -> TN={tn} FP={fp} FN={fn} TP={tp}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay(confusion_matrix(covid_true, covid_pred),
                           display_labels=["non-COVID", "COVID"]).plot(ax=ax[0])
    ax[0].set_title(f"{name}: confusion matrix")
    fpr, tpr, _ = roc_curve(covid_true, y_prob)
    ax[1].plot(fpr, tpr, label=f"AUC = {roc_auc_score(covid_true, y_prob):.3f}")
    ax[1].plot([0, 1], [0, 1], "--", color="gray")
    ax[1].set(xlabel="False positive rate", ylabel="True positive rate",
              title=f"{name}: ROC curve")
    ax[1].legend()
    plt.savefig(f"outputs/{name}_eval.png", dpi=150, bbox_inches="tight")
    print(f"Saved plot to outputs/{name}_eval.png")


if __name__ == "__main__":
    main()