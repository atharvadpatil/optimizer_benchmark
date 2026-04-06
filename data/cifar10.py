"""CIFAR-10 data loaders with standard augmentation."""

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir="./data_cache"):
    """Return (train_loader, test_loader) for CIFAR-10.

    Train: random crop 32 with padding 4 + horizontal flip + normalize.
    Test: normalize only.

    Args:
        batch_size: Batch size for both loaders.
        num_workers: DataLoader workers. Set to 0 if you hit macOS multiprocessing issues.
        data_dir: Directory to download/cache CIFAR-10.
    """
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader
