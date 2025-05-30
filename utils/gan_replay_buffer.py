import random
import torch

class GANReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def push(self, fake_canvas, target):
        # fake_canvas, target: [3, H, W] (单步)
        target = target.squeeze(0)
        self.buffer.append((fake_canvas.cpu(), target.cpu()))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        fake_canvases, targets = zip(*batch)
        fake_canvases = torch.stack(fake_canvases)
        targets = torch.stack(targets)
        return fake_canvases, targets

    def __len__(self):
        return len(self.buffer)