import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def get_loaders(root="data/raw", img_size=224, batch_size=32, seed=42):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full = datasets.ImageFolder(root)
    idx = np.random.RandomState(seed).permutation(len(full))
    n_train, n_val = int(0.7 * len(idx)), int(0.15 * len(idx))
    splits = [idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]]

    loaders = []
    for i, (tf, s) in enumerate(zip([train_tf, eval_tf, eval_tf], splits)):
        ds = Subset(datasets.ImageFolder(root, transform=tf), s)
        loaders.append(DataLoader(ds, batch_size=batch_size, shuffle=(i == 0), num_workers=0))
    return loaders, full.classes   # (train, val, test), class names