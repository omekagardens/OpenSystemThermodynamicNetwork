import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

def load_tiny_mnist(batch_size=64, subset_size=5000, label_noise=0.0, num_classes=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_full  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # Downsample train set
    indices = torch.randperm(len(train_full))[:subset_size]
    train = Subset(train_full, indices)

    # --- Inject label noise into the training subset ---
    if label_noise > 0.0:
        # We need to mutate the targets of the underlying dataset
        # train.dataset is the original MNIST dataset
        orig_targets = train.dataset.targets  # tensor of shape [60000]

        # indices in the full dataset we actually use
        noisy_idx_mask = torch.rand(len(indices)) < label_noise
        noisy_indices = indices[noisy_idx_mask]

        for idx in noisy_indices:
            true_label = orig_targets[idx].item()
            # pick a wrong label
            new_label = torch.randint(0, num_classes, (1,)).item()
            while new_label == true_label:
                new_label = torch.randint(0, num_classes, (1,)).item()
            orig_targets[idx] = new_label

        # Assign back (not strictly necessary, but keeps intention clear)
        train.dataset.targets = orig_targets

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_full, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader