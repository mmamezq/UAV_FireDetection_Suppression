import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import random
from torchvision import transforms, utils


class FireDataset(Dataset):

    def __init__(self, annotations_file, img_file, transform=None,
                 target_transform=None, joint_transform=None):

        with open(annotations_file, 'rb') as handle:
            self.img_labels = pickle.load(handle)


        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform
        self.joint_transform = joint_transform


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        with open(self.img_file, 'rb') as handle:
            images = pickle.load(handle)
        image = images[idx][1]
        label = self.img_labels[idx][1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        return image, label


class Normalize(object):

    def __init__(self,data_mean,std_dev):
        self.scale = random.uniform(0.01, 1)


    def __call__(self, image):
        image = (image - self.data_mean) / self.std_dev

        return image


def mean_std(loader):
    images, labels = next(iter(loader))


    # shape of images = [b,c,w,h]
    mean, std = images.mean([0, 2, 3]), images.std([0, 2, 3])

    return mean, std


def batch_mean_and_sd(loader):
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)
    for images, _ in loader:
        b, c, h, w = images.shape
    nb_pixels = b * h * w
    sum_ = torch.sum(images, dim=[0, 2, 3])
    sum_of_square = torch.sum(images ** 2,
                              dim=[0, 2, 3])

    fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
    snd_moment = (cnt * snd_moment + sum_of_square) / (cnt +

                                                       nb_pixels)

    cnt += nb_pixels
    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment **
                                       2)
    return mean, std
