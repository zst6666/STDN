import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import random
from torchvision import transforms as T

def load_NS(root):
    # Load SEVIR dataset
    filename = 'NS.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    return dataset

class NS(Dataset):
    def __init__(self, root, is_train=True, n_frames_input=10, n_frames_output=20, transform=None):
        super(NS, self).__init__()
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.dataset = load_NS(root)
        self.length = self.dataset.shape[0]
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.image_size_ = 64
        self.step_length_ = 0.1
        self.mean = 0
        self.std = 1

    def get_random_trajectory(self, seq_length):
        canvas_size = self.image_size_
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length_
            x += v_x * self.step_length_

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        length = self.n_frames_input + self.n_frames_output

        sequence = self.dataset[idx, ...]

        # Normalize the data to [0, 1] range
        sequence = sequence.astype(np.float32) / (1.0)

        input_frames = sequence[:self.n_frames_input]
        output_frames = sequence[self.n_frames_input:length] if self.n_frames_output > 0 else []

        input_tensor = torch.from_numpy(input_frames).contiguous().float()

        if output_frames.size > 0:
            output_tensor = torch.from_numpy(output_frames).contiguous().float()
        else:
            output_tensor = []

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(1)
            output_tensor = output_tensor.unsqueeze(1) if output_frames else []

        return input_tensor, output_tensor


def load_data(batch_size, val_batch_size, data_root, num_workers):
    file_path = data_root + '/NS/NS.npy'

    train_set = NS(root=data_root, is_train=True,
                      n_frames_input=10, n_frames_output=20)
    test_set = NS(root=data_root, is_train=False,
                     n_frames_input=10, n_frames_output=20)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std