# Few-shot dataloaders for UCM remote sensing dataset.

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
from abc import abstractmethod
from torchvision.datasets import ImageFolder

from PIL import ImageFile

from config_datasets import get_dataset_config

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASET_NAME = "UCM"
DATA_INFO = get_dataset_config(DATASET_NAME)
IMAGE_ROOT = DATA_INFO["images"]


def identity(x):
    return x


def _pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


class SimpleDataset:
    def __init__(self, transform, target_transform=identity):
        if not os.path.isdir(IMAGE_ROOT):
            raise FileNotFoundError(f"UCM images not found at {IMAGE_ROOT}")
        self.transform = transform
        self.target_transform = target_transform
        
        # Custom ImageFolder that excludes backup directories
        def is_valid_file(path):
            # Exclude backup and special directories
            path_parts = path.split(os.sep)
            for part in path_parts:
                if part in ['backup_original_splits', 'splits', '__pycache__', 'all_data.json']:
                    return False
            return True
        
        self.dataset = ImageFolder(IMAGE_ROOT, loader=_pil_loader, is_valid_file=is_valid_file)

    def __getitem__(self, i):
        path, label = self.dataset.samples[i]
        img = self.transform(_pil_loader(path))
        target = self.target_transform(label)
        return img, target

    def __len__(self):
        return len(self.dataset.samples)


class SetDataset:
    def __init__(self, batch_size, transform):
        if not os.path.isdir(IMAGE_ROOT):
            raise FileNotFoundError(f"UCM images not found at {IMAGE_ROOT}")

        dataset = ImageFolder(IMAGE_ROOT, loader=_pil_loader)
        self.classes = dataset.classes
        self.cl_list = list(range(len(self.classes)))

        self.sub_meta = {cl: [] for cl in self.cl_list}
        for img_path, label in dataset.samples:
            self.sub_meta[label].append(img_path)

        self.sub_dataloader = []
        sub_data_loader_params = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        for cl in self.cl_list:
            sub_dataset = SubDataset(self.sub_meta[cl], cl, transform=transform)
            self.sub_dataloader.append(
                torch.utils.data.DataLoader(sub_dataset, **sub_data_loader_params)
            )

    def __getitem__(self, i):
        return next(iter(self.sub_dataloader[i]))

    def __len__(self):
        return len(self.sub_dataloader)


class SubDataset:
    def __init__(self, sub_meta, cl, transform=transforms.ToTensor(), target_transform=identity):
        self.sub_meta = sub_meta
        self.cl = cl
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        img = self.transform(_pil_loader(self.sub_meta[i]))
        target = self.target_transform(self.cl)
        return img, target

    def __len__(self):
        return len(self.sub_meta)


class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


class TransformLoader:
    def __init__(
        self,
        image_size,
        normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4),
    ):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == "ImageJitter":
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        if transform_type == "Scale":
            return transforms.Resize([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        method = getattr(transforms, transform_type)
        if transform_type == "RandomSizedCrop":
            return method(self.image_size)
        elif transform_type == "CenterCrop":
            return method(self.image_size)
        elif transform_type == "Normalize":
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ["RandomSizedCrop", "ImageJitter", "RandomHorizontalFlip", "ToTensor", "Normalize"]
        else:
            transform_list = ["Scale", "CenterCrop", "ToTensor", "Normalize"]

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class DataManager(object):
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(transform)

        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader


class SetDataManager(DataManager):
    def __init__(self, image_size, n_way=5, n_support=5, n_query=16, n_eposide=100):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_eposide = n_eposide

        self.trans_loader = TransformLoader(image_size)

    def get_data_loader(self, aug):
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_eposide)
        data_loader_params = dict(batch_sampler=sampler, num_workers=4, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader


if __name__ == "__main__":
    pass
