import os

import lightning as pl
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST, ImageFolder


def get_data_module(cfg):
    """Factory function to instantiate a data module from its config."""
    print("Loading data module for dataset:", cfg.dataset.name)
    print("Data directory:", cfg.data_dir)
    if cfg.dataset.name == "mnist":
        return MNISTDataModule(cfg)
    elif cfg.dataset.name == "imagenet":
        return ImageNetDataModule(cfg)


def image_scaler(data):
    """Assumes data is in [0, 1] range and scales to [-1, 1]."""
    return (data - 0.5) * 2


def inverse_image_scaler(data):
    """Inverse of image_scaler, scales from [-1, 1] back to [0, 1]."""
    return (data / 2) + 0.5


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.batch_size = cfg.trainer.batch_size
        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(self.scaler),  # Scale to [-1, 1]
            ]
        )
        self.dims = (1, 28, 28)

    def prepare_data(self):  # only called on main process
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):  # called on every process
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir,
                train=False,
                transform=self.transform,
                download=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.mnist_test, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.data_dir = cfg.data_dir
        self.filter_for_classes = cfg.get("filter_for_classes", False)
        self.class_indices = cfg.get("filter_class_indices", [])
        self.batch_size = cfg.trainer.batch_size
        self.num_workers = cfg.trainer.num_workers
        self.resolution = cfg.dataset.img_resolution

        self.scaler = image_scaler
        self.inverse_scaler = inverse_image_scaler
        # Standard ImageNet normalization, equivalent to scaling
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resolution),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                # Scale to [-1, 1]
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.dims = (3, self.resolution, self.resolution)

    def _filter_by_classes(self, dataset):
        """
        Filter ImageFolder dataset by a set of class indices.
        """
        targets = dataset.targets  # list[int]
        indices = [i for i, y in enumerate(targets) if y in self.class_indices]
        return Subset(dataset, indices)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_dir = os.path.join(self.data_dir, "train")
            val_dir = os.path.join(self.data_dir, "val")
            self.imagenet_train = ImageFolder(train_dir, transform=self.transform)
            self.imagenet_val = ImageFolder(val_dir, transform=self.transform)
            if self.filter_for_classes and self.class_indices:
                self.imagenet_train = self._filter_by_classes(self.imagenet_train)
                print(f"Filtering validation data for classes: {self.class_indices}")
                self.imagenet_val = self._filter_by_classes(self.imagenet_val)

        if stage == "test" or stage is None:
            val_dir = os.path.join(self.data_dir, "val")
            # The validation set is commonly used as the test set for ImageNet
            self.imagenet_test = ImageFolder(val_dir, transform=self.transform)
            if self.filter_for_classes and self.class_indices:
                print(f"Filtering test data for classes: {self.class_indices}")
                self.imagenet_test = self._filter_by_classes(self.imagenet_test)

    def train_dataloader(self):
        return DataLoader(
            self.imagenet_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.imagenet_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.imagenet_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
