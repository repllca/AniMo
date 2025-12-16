# moving_mnist.py (very simple)
import numpy as np
import torch
from torchvision.datasets import MNIST
import torchvision.transforms as T

class MovingMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, mnist_root, seq_len=16, image_size=64, num_digits=2, train=True):
        self.mnist = MNIST(mnist_root, train=train, download=True, transform=T.ToTensor())
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_digits = num_digits

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        # generate one sequence by picking num_digits MNIST images and moving them
        imgs = []
        canvas = np.zeros((self.seq_len, self.image_size, self.image_size), dtype=np.float32)
        for d in range(self.num_digits):
            digit, _ = self.mnist[np.random.randint(0, len(self.mnist))]
            digit = digit.numpy().squeeze()  # 28x28
            # random initial pos and velocity
            x = np.random.randint(0, self.image_size-28)
            y = np.random.randint(0, self.image_size-28)
            vx = np.random.choice([-1,1]) * np.random.randint(1,3)
            vy = np.random.choice([-1,1]) * np.random.randint(1,3)
            for t in range(self.seq_len):
                x = np.clip(x + vx, 0, self.image_size-28)
                y = np.clip(y + vy, 0, self.image_size-28)
                canvas[t, y:y+28, x:x+28] += digit
        canvas = np.clip(canvas, 0, 1)
        # return tensor: C=1, T, H, W
        return torch.from_numpy(canvas).unsqueeze(0)  # 1, T, H, W
