import torch
from model import get_model

if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)  # fake batch of 4 images

    for name in ["cnn", "resnet"]:
        model = get_model(name)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output shape {tuple(out.shape)}, {n_params:,} parameters")