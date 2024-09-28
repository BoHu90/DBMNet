from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from torchvision import transforms

# pd.set_option('display.max_columns', 10000)
#
# pd.set_option('display.width', 10000)
#
# pd.set_option('display.max_colwidth', 10000)
# np.set_printoptions(threshold=np.inf)

# transform = transforms.Compose([
#     transforms.Resize((512, 512)),  # 将图片设置为（512，512）的尺寸
#     transforms.ToTensor(),  # 图片转张量，同时归一化0-255 ---》 0-1
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 减均值，除方差
# ])
#

class ImageDataset(Dataset):
    """Images dataset."""

    def __init__(self, root_dir, file_dir, transform=None):
        """
        Args:
             file_dir (string): Path to the txt file .
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.txt_data = np.genfromtxt(file_dir, delimiter=' ', dtype='str')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.txt_data)

    def __getitem__(self, idx):
        # try:
        restoreImage_name = str(os.path.join(self.root_dir, str(self.txt_data[idx, 0])))
        degradeImage_name = str(os.path.join(self.root_dir, str(self.txt_data[idx, 1])))

        D_mos = self.txt_data[idx, 2]
        D_mos = np.array(D_mos, dtype='float32')
        D_mos = torch.from_numpy(D_mos)
        degradeImage = Image.open(degradeImage_name).convert('RGB')
        restoreImage = Image.open(restoreImage_name).convert('RGB')
        # degradeImage.show()
        # restoreImage.show()
        if restoreImage.mode == 'P' or degradeImage.mode == 'P':
            degradeImage = degradeImage.convert('RGB')
            restoreImage = restoreImage.convert('RGB')
        # sample = {'degradeImage': degradeImage, 'restoreImage': restoreImage}  # 用字典存
        if self.transform:
            degradeImage = self.transform(degradeImage)
            restoreImage = self.transform(restoreImage)

        return [degradeImage, restoreImage, D_mos]



# root_dir = '/usr/Python_Project/dataset/MD_Uniform_Database'
# file_dir = './test_pair.txt'


if __name__ == '__main__':
    # data = ImageDataset(root_dir, file_dir, transform)[9]
    # data = np.array(data)
    # print(data.shape)
    pass
