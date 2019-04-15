import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class UCSDPedDataset(Dataset):
    def __init__(self, root_dir, train=True, frame_length=10, frame_stride=1, frame_dilation=0, transforms=None, dimension=3):
        self.train = train
        self.root_dir = root_dir
        self.prefix = 'Train' if self.train else 'Test'
        self.dir = os.path.join(self.root_dir, self.prefix)
        self.frame_length = frame_length
        self.frame_stride = frame_stride
        self.frame_dilation = frame_dilation
        self.transforms = transforms
        self.dimension = dimension
        self.list = self._getlist()
    
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        folder_num, indices = self.list[index]
        imgs = []
        for idx in indices:
            img = Image.open(os.path.join(self.dir, '{}{:03d}'.format(self.prefix, folder_num + 1), '{:03d}.tif'.format(idx + 1)))
            if self.transforms is not None:
                img = self.transforms(img)

            imgs.append(img)
        
        return torch.stack(imgs, dim=1) if self.dimension == 3 else torch.cat(imgs, dim=0), folder_num

    def _getlist(self):
        max_folder = len(os.listdir(self.dir))
        lists = []
        for i in range(max_folder):
            path = os.path.join(self.dir, '{}{:03d}'.format(self.prefix, i + 1))
            max_images = len(os.listdir(path))
            for j in self._getstartindices(max_images):
                lists.append((i, self._getindices(j)))
        return lists
    
    def _getstartindices(self, max_images):
        return list(range(0, max_images - self._getdilatedlength() + 1, self.frame_stride))

    def _getdilatedlength(self):
        return self.frame_length * (self.frame_dilation + 1) - self.frame_dilation

    def _getindices(self, start_id):
        return list(range(start_id, start_id + self._getdilatedlength(), self.frame_dilation + 1))

class UCSDPedFlowDataset(UCSDPedDataset):
    def __init__(self, root_dir, flow_root, train=True, frame_length=2, frame_stride=1, frame_dilation=0, transforms=None, dimension=3):
        super().__init__(root_dir, train, frame_length, frame_stride, frame_dilation, transforms, dimension)
        self.flow_root = flow_root
        self.flow_dir = os.path.join(self.flow_root, self.prefix)

    def __getitem__(self, index):
        folder_num, indices = self.list[index]
        imgs = []
        for idx in indices:
            img = Image.open(os.path.join(self.dir, '{}{:03d}'.format(self.prefix, folder_num + 1), '{:03d}.tif'.format(idx + 1)))
            if self.transforms is not None:
                img = self.transforms(img)

            imgs.append(img)
        flows = self._getflow(indices[0], folder_num)
        
        return torch.stack(imgs, dim=1) if self.dimension == 3 else torch.cat(imgs, dim=0), flows, folder_num

    def _getflow(self, start_id, folder_num):
        filename = os.path.join(self.flow_dir, '{}{:03d}'.format(self.prefix, folder_num + 1), '{:06d}.flo'.format(start_id))
        return read_flow(filename)

def read_flow(filename):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))