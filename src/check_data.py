from pathlib import Path

root = Path("data/raw")
for cls in sorted(root.iterdir()):
    if cls.is_dir():
        n = len([f for f in cls.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        print(f"{cls.name}: {n} images")