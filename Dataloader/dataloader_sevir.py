# import torch
# import numpy as np
# import torch.utils.data as data
# from torch.utils.data import Dataset,Subset
# class SEVIR(data.Dataset):
#     def __init__(self, file_path, transform=None, t_slice=None):
#         self.data = np.load(file_path)
#         self.transform = transform
#         self.num_samples, self.time_steps, self.channels, self.height, self.width = self.data.shape
#         self.t_slice = t_slice if t_slice is not None and t_slice <= self.time_steps else self.time_steps
#
#     def __len__(self):
#         return self.num_samples  # 返回数据集中的样本数量
#
#     def __getitem__(self, idx):
#         # 获取对应索引的样本数据（整个时间序列或切片后的时间序列）
#         sample = self.data[idx, :self.t_slice]
#
#         # 如果需要，可以在这里应用transform（注意：这需要对整个时间序列应用transform）
#         if self.transform:
#             # 注意：这里的transform应该是对numpy数组操作的，因为数据还没有转换为torch张量
#             sample = self.transform(sample)
#
#             # 将numpy数组转换为torch张量
#         sample = torch.from_numpy(sample).float()
#
#         # 标准化数据（在每个样本的维度上计算均值和标准差）
#         mean = torch.mean(sample, dim=(0, 2, 3))  # 计算时间步长、高度和宽度的均值（保持通道维度不变）
#         std = torch.std(sample, dim=(0, 2, 3))  # 计算时间步长、高度和宽度的标准差（保持通道维度不变）
#
#         # 避免除以零
#         std = torch.where(std > 1e-5, std, torch.ones_like(std))
#
#         # 标准化数据（注意：这里要保持通道维度与均值和标准差的维度一致）
#         standardized_sample = (sample - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
#
#         # 返回切片后的时间序列数据，形状为(C, T_slice, H, W)
#         standardized_sample = standardized_sample.permute(0, 1, 2, 3)  # 转换维度顺序以匹配(C, T_slice, H, W)
#
#         # 检查标准化是否正确应用
#         if self.transform is None:  # 只有在没有应用其他transform时才检查
#             mean_check = torch.mean(standardized_sample)
#             std_check = torch.std(standardized_sample)
#             # print(f"Sample {idx}: Mean {mean_check.item()}, Std {std_check.item()}")
#
#         return standardized_sample  # 转换维度顺序以匹配(C, T_slice, H, W)
#
# def load_data(batch_size,val_batch_size,data_root,num_workers):
#     file_path = 'SEVIR_ir069.npy'
#     t_slice = 10
#     sevir_dataset = SEVIR(file_path, t_slice=t_slice)
#
#     train_size = int(0.7 * len(sevir_dataset))  # 使用70%的数据作为训练集
#     test_size = len(sevir_dataset) - train_size  # 剩余的30%作为测试集
#
#     # 使用torch.utils.data.Subset来划分数据集
#     train_indices = range(train_size)
#     test_indices = range(train_size, train_size + test_size)
#     train_dataset = Subset(sevir_dataset, train_indices)
#     test_dataset = Subset(sevir_dataset, test_indices)
import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import random
from torchvision import transforms as T

# filename = 'SEVIR_ir069.npy'
def load_sevir(root):
    # Load SEVIR dataset
    filename = 'SEVIR_ir069.npy'
    path = os.path.join(root, filename)
    dataset = np.load(path)
    return dataset
class SEVIR(Dataset):
    def __init__(self,root,is_train=True,n_frames_input=10,n_frames_output=10,transform=None):
        super(SEVIR,self).__init__()
        self.is_train = is_train
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output
        self.transform = transform
        self.dataset = load_sevir(root)
        self.length = self.dataset.shape[0]
        self.n_frames_total = self.n_frames_input + self.n_frames_output
        self.image_size_ = 129
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
        # Assuming each sequence in the .npy file has a length of n_frames_input + n_frames_output
        length = self.n_frames_input + self.n_frames_output

        # Load the data for the given index
        sequence = self.dataset[idx, ...]  # Adjust this based on the actual shape of your .npy data

        # Normalize the data to [0, 1] range
        sequence = sequence.astype(np.float32) / 255.0

        # Split the sequence into input and output frames
        input_frames = sequence[:self.n_frames_input]
        output_frames = sequence[self.n_frames_input:length] if self.n_frames_output > 0 else []

        # Convert the numpy arrays to PyTorch tensors
        input_tensor = torch.from_numpy(input_frames).contiguous().float()
        output_tensor = torch.from_numpy(output_frames).contiguous().float() if output_frames else []

        # If needed, add an extra dimension for channels (assuming grayscale images)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(1)
            output_tensor = output_tensor.unsqueeze(1) if output_frames else []

        return input_tensor, output_tensor


def load_data(batch_size, val_batch_size, data_root, num_workers):
    file_path = data_root + '/SEVIR/SEVIR_ir069.npy'

    train_set = SEVIR(root=data_root, is_train=True,
                            n_frames_input=10, n_frames_output=10)
    test_set = SEVIR(root=data_root, is_train=False,
                           n_frames_input=10, n_frames_output=10)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std

    # train_size = int(0.7 * len(sevir_dataset))
    # test_size = len(sevir_dataset) - train_size
    #
    # train_indices = list(range(train_size))
    # test_indices = list(range(train_size, len(sevir_dataset)))
    #
    # train_dataset = Subset(sevir_dataset, train_indices)
    # test_dataset = Subset(sevir_dataset, test_indices)
    #
    # # Assuming you want to use DataLoaders for batch processing
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
    #                                            num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False,
    #                                           num_workers=num_workers)
    # vail_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False,
    #                                           num_workers=num_workers)
    # mean, std = 0, 1
    # return train_loader, test_loader,vail_loader,mean,std

#
# import os
# import gzip
# import random
# import numpy as np
# import torch
# import torch.utils.data as data
#
#
# # 假设 load_mnist 和 load_fixed_set 函数已经在前面定义过了
# # ...
#
# class MovingMNIST(data.Dataset):
#     # 构造函数和其他方法保持不变
#     # ...
#
#     def __getitem__(self, idx):
#         # 生成或加载数据的代码保持不变
#         # ...
#
#         # 修改数据形状以符合 (time_steps, channels, height, width)
#
#
#     # 修改 load_data 函数以适应新的数据形状
#
#
# def load_data(batch_size, val_batch_size, data_root, num_workers):
#     # 创建数据集实例的代码保持不变
#     # ...
#
#     # 修改 DataLoader 以处理新的数据形状
#     def collate_fn(batch):
#         inputs, outputs = zip(*batch)
#         # 合并批处理维度和时间维度，然后重新排列以匹配 (batch_size, time_steps, channels, height, width)
#         inputs = torch.cat(inputs, dim=0).permute(1, 0, 2, 3, 4).contiguous()
#         if outputs[0].size > 0:  # 检查输出是否为空（可能在某些情况下不使用输出）
#             outputs = torch.cat(outputs, dim=0).permute(1, 0, 2, 3, 4).contiguous()
#         return inputs, outputs
#
#     dataloader_train = torch.utils.data.DataLoader(
#         train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers, collate_fn=collate_fn)
#     dataloader_validation = torch.utils.data.DataLoader(
#         test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
#         collate_fn=collate_fn)
#     dataloader_test = torch.utils.data.DataLoader(
#         test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
#         collate_fn=collate_fn)
#
#     mean, std = 0, 1  # 这些值应该根据数据集计算，这里仅作为示例
#     return dataloader_train, dataloader_validation, dataloader_test, mean, std