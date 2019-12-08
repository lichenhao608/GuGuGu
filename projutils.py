# -------------------------------------------------------------------------- #
# Follow the tutorial of creating Dataloader of PyTorch                      #
# Website: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html #
# -------------------------------------------------------------------------- #

from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

train_path = 'data/train/'
label_file = 'data/train_labels.csv'


class CellImgDataset(Dataset):
    '''Cell image dataset'''

    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the training images.
            transform (callable, optional): Optional transform to be applied on
                a sample.
        '''
        self.cellimg_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.cellimg_frame)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        img_name = os.path.join(
            self.root_dir, self.cellimg_frame.iloc[index, 0]+'.tif')
        image = io.imread(img_name)
        label = self.cellimg_frame.iloc[index, 1]
        label = np.array(label)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    '''Rescale the image in a sample to a given size'''

    def __init__(self, output_size):
        '''
        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched to
                output_size keeping aspect ratio the same.
        '''
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


class RandomCrop(object):
    '''Crop randomly the image in a sample'''

    def __init__(self, output_size):
        '''
        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        '''
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:(top + new_h), left:(left + new_w)]

        return {'image': imaga, 'label': label}


class RandomHorizontalFlip(object):
    '''Randomly flip image horizontally'''

    def __init__(self, p=0.5):
        '''
        Args:
            p (float, optional): The probability that image will flip
        '''
        assert 0 <= p < 1
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.flip(image, axis=1).copy()

        return {'image': image, 'label': label}


class ToTensor(object):
    '''Convert ndarrays in sample to Tensors.'''

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).long()}


def train_test_loader(csv_file, root_dir, seed=0, batch_size=4, shuffle=True,
                      num_workers=4, pin_memory=False, transform=None, train_size=1.0):
    '''Get training and testing dataloader from the data
        Original Source from @kevinzakka
        https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the training images.
        seed (int, optional): Set seed for redoable result.
        batch_size (int, optional): How many samples per batch to load.
        shuffle (bool, optional): whether to shuffle the train/test indices.
        num_workers (int, optional): number of subprocesses to use when loading the
        dataset.
        pin_memory (bool, optional):  If True, the data loader will copy Tensors
            into CUDA pinned memory before returning them. 
        transform (callable, optional): Optional transform to be applied on
            a sample.
        train_size (float, optional): Train set size in range (0, 1]. 1 means no
            test set. Default is 1.

    Returns:
        train_loader (Dataloader): Training set Dataloader
        test_loader (Dataloader): Testing set Dataloader. Retuen None if
            train_size is 1.
    '''
    assert 0.0 < train_size <= 1.0

    train_dataset = CellImgDataset(csv_file, root_dir, transform=transform)
    test_dataset = CellImgDataset(
        csv_file, root_dir, transform=transforms.Compose([ToTensor()]))

    N = len(train_dataset)
    indices = np.arange(N)
    split = int(np.floor(train_size * N))

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    train_ind, test_ind = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_ind)
    if train_size < 1:
        test_sampler = SubsetRandomSampler(test_ind)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_memory
    )
    if train_size < 1:
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size,
            sampler=test_sampler, num_workers=num_workers,
            pin_memory=pin_memory
        )
    else:
        test_loader = None

    return train_loader, test_loader


def load_data(label_file, img_dir, num=None):
    '''Convert image file into numpy array

    Args:
        label_file: file directory to the csv file containing label info
        img_dir: repository containing images
        num: number of file to load, if None, all files will be loaded, defualt
            None
    Return:
        images: (N, 96, 96) shape image file
        y: (N,) shape label info
    '''

    label = pd.read_csv(label_file)
    y = label['label'].values
    N = len(label)

    if not num:
        num = N

    images = np.zeros((num, 96, 96, 3))

    for i, name in enumerate(label['id'].values):
        img_file = img_dir + name + '.tif'
        img = plt.imread(img_file)
        images[i] = img / 255
        # convert image to grayscale
        # images[i] = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        if i % 2000 == 0:
            print(f'process {i} images')

        if i + 1 == num:
            break

    return images, y[:num]
