import sys
import json
import torch
import torch.nn as nn
from tqdm import tqdm

from data import get_loaders
from model import get_model

EPOCHS = 20


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "cnn"  # "cnn" or "resnet"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training '{name}' on {device}")

    (train_dl, val_dl, _), classes = get_loaders()
    model = get_model(name).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 1e-3 if name == "cnn" else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    def run_epoch(dl, train):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        with torch.set_grad_enabled(train):
            for x, y in tqdm(dl, leave=False):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item() * len(y)
                correct += (out.argmax(1) == y).sum().item()
                total += len(y)
        return loss_sum / total, correct / total

    history, best_val = [], float("inf")
    for epoch in range(EPOCHS):
        tr_loss, tr_acc = run_epoch(train_dl, True)
        va_loss, va_acc = run_epoch(val_dl, False)
        scheduler.step(va_loss)
        history.append(dict(epoch=epoch + 1, tr_loss=tr_loss, tr_acc=tr_acc,
                            va_loss=va_loss, va_acc=va_acc))
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), f"outputs/{name}_best.pt")

    with open(f"outputs/{name}_history.json", "w") as f:
        json.dump(history, f)
    print("Done. Best model saved to", f"outputs/{name}_best.pt")


if __name__ == "__main__":
    main()