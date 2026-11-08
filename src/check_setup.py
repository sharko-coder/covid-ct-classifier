import torch, torchvision, sklearn, cv2

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())