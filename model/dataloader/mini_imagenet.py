import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH2, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')


def identity(x):
    return x


class MiniImageNet(Dataset):
    def __init__(self, setname, args, augment=False):

        self.setname = setname
        self.dataset_dir = '/home/ubuntu/code/miniImageNet_COSOC/images'
        self.data_dir = os.path.join(self.dataset_dir, self.setname)
        self.args = args

        cat_container = sorted(os.listdir(self.data_dir))
        cats2label = {cat: label for label, cat in enumerate(cat_container)}

        dataset = []
        labels = []
        for cat in cat_container:
            for img_path in sorted(os.listdir(os.path.join(self.data_dir, cat))):
                if '.jpg' not in img_path:
                    continue
                label = cats2label[cat]
                dataset.append((os.path.join(self.data_dir, cat, img_path)))
                labels.append(label)

        self.data, self.label = dataset, labels
        self.num_class = len(set(self.label))

        self.patch_list = [1, args.num_edge]
        self.patch_ratio = 1
        image_size = 84
        self.num_patch = args.num_patch
        self.image_size = 84

        self.transform_rand = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                 np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
        ])

        if setname == 'val' or setname == 'test':
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                     np.array([0.2726, 0.2634, 0.2794]))])
        elif setname == 'train':
            self.transform = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.4712, 0.4499, 0.4031]),
                                     np.array([0.2726, 0.2634, 0.2794]))
            ])

        else:
            raise ValueError('no such set')

    def __len__(self):
        return len(self.data)

    def get_grid_location(self, size, ratio, num_grid):
        '''

        :param size: size of the height/width
        :param ratio: generate grid size/ even divided grid size
        :param num_grid: number of grid
        :return: a list containing the coordinate of the grid
        '''
        raw_grid_size = int(size / num_grid)
        enlarged_grid_size = int(size / num_grid * ratio)

        center_location = raw_grid_size // 2

        location_list = []
        for i in range(num_grid):
            location_list.append((max(0, center_location - enlarged_grid_size // 2),
                                  min(size, center_location + enlarged_grid_size // 2)))
            center_location = center_location + raw_grid_size

        return location_list

    def get_pyramid(self, img, num_patch):
        if self.setname == 'val' or self.setname == 'test':
            num_grid = num_patch
            grid_ratio = self.patch_ratio

        elif self.setname == 'train':
            num_grid = num_patch
            grid_ratio = self.patch_ratio
            # grid_ratio=1+2*random.random()
        else:
            raise ValueError('Unkown set')

        w, h = img.size
        grid_locations_w = self.get_grid_location(w, grid_ratio, num_grid)
        grid_locations_h = self.get_grid_location(h, grid_ratio, num_grid)

        patches_list = []
        for i in range(num_grid):
            for j in range(num_grid):
                patch_location_w = grid_locations_w[j]
                patch_location_h = grid_locations_h[i]
                left_up_corner_w = patch_location_w[0]
                left_up_corner_h = patch_location_h[0]
                right_down_cornet_w = patch_location_w[1]
                right_down_cornet_h = patch_location_h[1]
                patch = img.crop((left_up_corner_w, left_up_corner_h, right_down_cornet_w, right_down_cornet_h))
                patch = self.transform(patch)
                patches_list.append(patch)

        return patches_list

    def __getitem__(self, i):  # return the ith data in the set.
        path, label = self.data[i], self.label[i]

        image = Image.open(path).convert('RGB')

        patch_list = []
        for num_patch in self.patch_list:
            patches = self.get_pyramid(image, num_patch)
            patch_list.extend(patches)

        for _ in range(self.num_patch):
            patch_list.append(self.transform_rand(image))

        patch_list = torch.stack(patch_list, dim=0)

        return patch_list, label
