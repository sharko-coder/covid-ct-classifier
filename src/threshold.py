import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from experiment import kaggle_splits, external_splits, predict
from model import get_model


def rates(y, p, t):
    covid = (y == 0).astype(int)
    pred = (p > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(covid, pred, labels=[0, 1]).ravel()
    return (tp + tn) / len(y), tp / (tp + fn), tn / (tn + fp)


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, _, k_te = kaggle_splits()
    _, e_va, e_te = external_splits()

    model = get_model("resnet").to(device)
    model.load_state_dict(torch.load("outputs/resnet_mixed_best.pt", map_location=device))

    # choose threshold on external VALIDATION data only
    yv, pv = predict(model, e_va, device)
    thresholds = np.linspace(0.05, 0.95, 91)
    ok = [t for t in thresholds if rates(yv, pv, t)[1] >= 0.90]
    if ok:
        t_star = max(ok)  # highest threshold still reaching 90% sensitivity
    else:  # fall back to Youden's J if 90% sensitivity is unreachable
        t_star = max(thresholds, key=lambda t: sum(rates(yv, pv, t)[1:]) - 1)
    print(f"Chosen threshold (external val): {t_star:.2f}\n")

    for name, samples in [("Kaggle test", k_te), ("External test", e_te)]:
        y, p = predict(model, samples, device)
        for t in (0.5, t_star):
            acc, sens, spec = rates(y, p, t)
            print(f"{name:14s} thr {t:.2f} | acc {acc:.3f} | sens {sens:.3f} | spec {spec:.3f}")


if __name__ == "__main__":
    main()