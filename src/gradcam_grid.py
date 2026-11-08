import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms

from model import get_model
from data import MEAN, STD

CLASSES = ["COVID", "non-COVID"]


def main():
    device = torch.device("cpu")
    model = get_model("resnet")
    model.load_state_dict(torch.load("outputs/resnet_best.pt", map_location=device))
    model.eval()

    feats, grads = {}, {}
    layer = model.layer4[-1]
    layer.register_forward_hook(lambda m, i, o: feats.update(a=o))
    layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    # Recreate the exact same test split as data.py
    root = "data/raw"
    full = datasets.ImageFolder(root)
    idx = np.random.RandomState(42).permutation(len(full))
    n_train, n_val = int(0.7 * len(idx)), int(0.15 * len(idx))
    test_idx = idx[n_train + n_val:]

    def explain(path):
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0)
        out = model(x)
        probs = torch.softmax(out, dim=1)[0]
        cls = out.argmax(1).item()
        model.zero_grad()
        out[0, cls].backward()
        w = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * feats["a"]).sum(1)).squeeze().detach().numpy()
        cam = cv2.resize(cam / (cam.max() + 1e-8), (224, 224))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)[:, :, ::-1]
        base = np.array(img.resize((224, 224)))
        overlay = (0.55 * base + 0.45 * heat).astype("uint8")
        return overlay, cls, probs[cls].item()

    # Collect: correct COVID, correct non-COVID, and mistakes
    correct_covid, correct_non, wrong = [], [], []
    for i in test_idx:
        path, label = full.samples[i]
        overlay, pred, conf = explain(path)
        item = (overlay, label, pred, conf)
        if pred != label:
            wrong.append(item)
        elif label == 0 and len(correct_covid) < 4:
            correct_covid.append(item)
        elif label == 1 and len(correct_non) < 4:
            correct_non.append(item)
        if len(correct_covid) == 4 and len(correct_non) == 4 and len(wrong) >= 6:
            break

    rows = [("Correct COVID", correct_covid),
            ("Correct non-COVID", correct_non),
            ("MISTAKES", wrong[:6])]
    fig, axes = plt.subplots(3, 6, figsize=(18, 9.5))
    for r, (title, items) in enumerate(rows):
        for c in range(6):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(items):
                overlay, label, pred, conf = items[c]
                ax.imshow(overlay)
                ax.set_title(f"true {CLASSES[label]}\npred {CLASSES[pred]} ({conf:.0%})",
                             fontsize=9)
        axes[r, 0].text(-0.1, 0.5, title, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/gradcam_grid.png", dpi=130, bbox_inches="tight")
    print(f"Saved outputs/gradcam_grid.png ({len(wrong)} mistakes found)")


if __name__ == "__main__":
    main()