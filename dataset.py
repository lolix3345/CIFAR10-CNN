import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pickle

class Cifar10Dataset(Dataset):
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform

        if split == "train":
            self.imgs, self.labels = self.obtain_train_set()
        elif split=="test":
            self.imgs, self.labels = self.obtain_test_set()
        else:
            print("split name not recognized : must be either \"train\" or \"test\"")
    def obtain_train_set(self):
        totensor = transforms.ToTensor()
        train_imgs = torch.zeros(50000, 3, 32, 32, dtype=torch.float)
        train_labels = torch.zeros(50000, dtype=torch.long)
        for i in range(0, 5):
            data1 = Cifar10Dataset.unpickle(os.path.join(self.root_dir, "data_batch_" + str(i + 1)))
            train_labels[(i * 10000):((i + 1) * 10000)] = torch.tensor(data1[b'labels'], dtype=torch.long)
            train_imgs[(i * 10000):((i + 1) * 10000), :, :, :] = totensor(data1[b'data']).resize(1, 10000, 3, 32, 32).squeeze()
        return train_imgs, train_labels

    def obtain_test_set(self):
        totensor = transforms.ToTensor()
        test_imgs = torch.zeros(10000, 3, 32, 32, dtype=torch.float)
        test_labels = torch.zeros(10000, dtype=torch.long)
        data1 = Cifar10Dataset.unpickle(os.path.join(self.root_dir, "test_batch"))
        test_labels[:] = torch.tensor(data1[b'labels'], dtype=torch.long)
        test_imgs[:, :, :, :] = totensor(data1[b'data']).resize(1, 10000, 3, 32, 32).squeeze()

        return test_imgs, test_labels

    def __len__(self):
        return self.imgs.size()[0]

    def __getitem__(self, item):
        images = self.imgs[item]
        labels = self.labels[item]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels
