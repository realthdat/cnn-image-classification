import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_root="data/intel", img_size=160, batch_size=64, num_workers=2):
    tfm_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*3, (0.5,)*3),
    ])

    train_dir = os.path.join(data_root, "seg_train")
    val_dir   = os.path.join(data_root, "seg_test")

    train_ds = datasets.ImageFolder(train_dir, transform=tfm_train)
    val_ds   = datasets.ImageFolder(val_dir, transform=tfm_eval)
    test_ds  = val_ds

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_ld, val_ld, test_ld, train_ds.classes
