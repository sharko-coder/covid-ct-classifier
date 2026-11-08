import sys
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from model import get_model
from data import MEAN, STD

CLASSES = ["COVID", "non-COVID"]


def main():
    img_path = sys.argv[1]
    out_name = sys.argv[2] if len(sys.argv) > 2 else "gradcam.png"
    device = torch.device("cpu")  # simplest for hooks

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
    img = Image.open(img_path).convert("RGB")
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

    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(base); ax[0].set_title("Original"); ax[0].axis("off")
    ax[1].imshow(overlay); ax[1].axis("off")
    ax[1].set_title(f"Predicted: {CLASSES[cls]} ({probs[cls]:.1%})")
    plt.savefig(f"outputs/{out_name}", dpi=150, bbox_inches="tight")
    print(f"Predicted {CLASSES[cls]} ({probs[cls]:.1%}). Saved to outputs/{out_name}")


if __name__ == "__main__":
    main()