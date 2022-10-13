import torch
import torchvision


def _load_from_dataset(dataset, rank, world_size, batch_size, pin_memory=True, shuffle=True, num_workers=0):

    sampler = torch.utils.data.distributed \
                .DistributedSampler(dataset,
                                    rank=rank,
                                    num_replicas=world_size,
                                    shuffle=shuffle)

    loader = torch.utils.data \
                .DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    return loader


def _load_from_dataset_nondistributed(dataset, rank, world_size, batch_size, pin_memory=True, shuffle=True, num_workers=0):

    sampler = torch.utils.data.RandomSampler(dataset)

    loader = torch.utils.data \
                .DataLoader(dataset,
                            batch_size=batch_size,
                            sampler=sampler,
                            pin_memory=pin_memory,
                            num_workers=num_workers)

    return loader


from torchvision.datasets import MNIST
class SortedMNIST(MNIST):
    def __init__(self, *args, rank=0, world_size=1, **kwargs):
        self.rank = rank
        self.world_size = world_size
        super().__init__(*args, **kwargs)
    def _load_data(self):
        data, targets = super()._load_data()
        if self.train:
            keys = targets.unique()
            max_samples = min([(targets == k).sum() for k in keys]).item()
            mask = targets == self.rank
            # order = targets.argsort()
            data = data[mask][:max_samples].detach()
            targets = targets[mask][:max_samples].detach()
        return data, targets


def load_mnist(rank, world_size, batch_size, shuffle=False, num_workers=0, sort=False):

    from torchvision.datasets import MNIST
    import torchvision.transforms as tf

    transform = tf.Compose([tf.ToTensor(),
                            tf.Normalize((0.1307,), (0.3081,))])
    download = rank == 0
    if not sort:
        train_data = MNIST('~/data', train=True, transform=transform, download=download)
        train_loader = _load_from_dataset(train_data, rank, world_size, batch_size, shuffle=shuffle, num_workers=num_workers)
    if sort:
        train_data = SortedMNIST('~/data', rank=rank, train=True, transform=transform, download=download)
        train_loader = _load_from_dataset_nondistributed(train_data, rank, world_size, batch_size, shuffle=shuffle, num_workers=num_workers)

    val_data = MNIST('~/data', train=False, transform=transform, download=download)

    val_loader = _load_from_dataset(val_data, 0, 1, batch_size, shuffle=False,
                                    num_workers=num_workers)

    return train_loader, val_loader
