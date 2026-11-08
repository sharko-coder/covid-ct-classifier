from data import get_loaders

if __name__ == "__main__":
    (train_dl, val_dl, test_dl), classes = get_loaders()

    print("Classes:", classes)
    print("Train / Val / Test images:",
          len(train_dl.dataset), len(val_dl.dataset), len(test_dl.dataset))

    x, y = next(iter(train_dl))
    print("Batch shape:", x.shape)
    print("Labels in batch:", y.tolist())