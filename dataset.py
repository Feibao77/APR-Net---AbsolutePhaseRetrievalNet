import os

import cv2
import torch
import torch.utils.data as data
import h5py
import numpy as np
from torchvision import transforms
import scipy.io as scio


class APRDataset(data.Dataset):
    def __init__(self, root: str, train: bool = True, transforms=None):
        assert os.path.exists(root), f"path '{root}' does not exist."
        if train:
            self.image_root = os.path.join(root, "Input cp2/train_Input cp2")
            self.mask_root = os.path.join(root, "Output cp2/train_Output_cp2")
        else:
            self.image_root = os.path.join(root, "Input cp2/val_Input cp2")
            self.mask_root = os.path.join(root, "Output cp2/val_Output cp2")
        assert os.path.exists(self.image_root), f"path '{self.image_root}' does not exist."
        assert os.path.exists(self.mask_root), f"path '{self.mask_root}' does not exist."

        self.image_names = [p for p in os.listdir(self.image_root) if p[0] == 's']
        self.mask_names = [p for p in os.listdir(self.mask_root) if p[0] == 'a']

        self.image_names.sort()
        self.mask_names.sort()
        print(self.image_names)
        print(self.mask_names)

        assert len(self.image_names) > 0, f"not find any images in {self.image_root}."

        self.images_path = [os.path.join(self.image_root, n) for n in self.image_names]
        self.masks_path = [os.path.join(self.mask_root, n) for n in self.mask_names]

        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.images_path[idx]
        mask_path = self.masks_path[idx]

        image = scio.loadmat(image_path)
        assert image is not None, f"failed to read image: {image_path}"

        image_temp = image['cp2']
        img = image_temp[:] / 255.
        image_tensor = torch.tensor(img, dtype=float,
                                    # device='cuda:0'
                                    )
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.float()

        c, h, w = image_tensor.shape
        target = np.empty((1, h, w), dtype='float32')

        target_temp = scio.loadmat(mask_path)
        assert target_temp is not None, f"failed to read mask: {mask_path}"
        target_temp = target_temp['pha123_actual']
        target[0, :, :] = target_temp[:]

        # target
        target_tensor = torch.tensor(target, dtype=float,
                                     # device='cuda:0'
                                     )
        target_tensor = target_tensor.float()

        return image_tensor, target_tensor

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)

        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


if __name__ == '__main__':
