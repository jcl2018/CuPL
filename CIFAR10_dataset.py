import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np

import pickle
from PIL import Image

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


CN_class_names = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
#CN_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class CIFAR10Dataset(Dataset):
    def __init__(self, CIFAR_root, transform=None):
        self.transform = transform
        # 1. Read metadata
        meta_dict = unpickle("%s/batches.meta" % CIFAR_root)
        self.num_data_per_batch = meta_dict["num_cases_per_batch".encode("utf-8")]
        class_names = meta_dict["label_names".encode("utf-8")]
        self.class_names = [s.decode("utf-8") for s in class_names]
        num_vis = meta_dict["num_vis".encode("utf-8")]

        self.num_batch = 5
        print("Total number of data is :", self.num_batch * self.num_data_per_batch)
        print("All (ordered) class names are: ", self.class_names)
        print(num_vis)  # 32 x 32 x 3 = 3072
        # 2. Read data batch

        self.all_data = np.zeros((self.num_batch * self.num_data_per_batch, 32, 32, 3))
        self.all_label = []
        for batch_idx in range(self.num_batch):
            data_dict = unpickle("%s/data_batch_%d" % (CIFAR_root, batch_idx + 1))

            data = data_dict["data".encode("utf-8")]
            data = data.reshape((len(data), 3, 32, 32))
            data = data.transpose(0, 2, 3, 1)
            self.all_data[batch_idx * self.num_data_per_batch: (batch_idx + 1) * self.num_data_per_batch] = data

            labels = data_dict["labels".encode("utf-8")]
            self.all_label += labels

        self.idx_to_label = CN_class_names
        self.idx_to_text = CN_class_names

    def __len__(self):
        return self.num_data_per_batch * self.num_batch

    def __getitem__(self, index):
        img = self.all_data[index].astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = self.all_label[index]
        # (img, label, index)
        return img, label, index


if __name__ == "__main__":

    # Create an instance of your dataset
    dataset = CIFAR10Dataset("dataset/cifar-10-python/cifar-10-batches-py")

    # Create a data loader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Iterate over the data loader
    for batch in dataloader:
        # Perform training or inference with the batch
        img, label, index = batch
        print(img.shape)
        break
