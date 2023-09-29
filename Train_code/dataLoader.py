import os
import torch
from torchvision import datasets, transforms
from DatasetSplitter import DatasetSplitter  

def load_data(data_dir, batch_size, num_workers, img_size, weighted_sampling=False, train_perc=0.8):
    """
    Load and preprocess the dataset.

    Args:
        data_dir (str): Path to the root directory of the dataset.
        batch_size (int): Batch size for DataLoader.
        num_workers (int): Number of data loading workers.
        img_size (int): Size of the input images after resizing.
        weighted_sampling (bool): Whether to use weighted sampling for imbalanced datasets.

    Returns:
        dataloaders (dict): Dictionary of DataLoader objects for 'train' and 'val' sets.
        dataset_sizes (dict): Dictionary containing the sizes of 'train' and 'val' sets.
        class_names (list): List of class names.
    """
    if train_perc < 1.0:
        splitter = DatasetSplitter(data_dir, os.path.join(data_dir, 'train'), os.path.join(data_dir, 'val'), train_perc)
        splitter.split_dataset()
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if weighted_sampling:
        counts = torch.bincount(torch.tensor(image_datasets['train'].targets))
        weights = 1.0 / counts.float()
        sample_weights = weights[image_datasets['train'].targets]
        sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, len(sample_weights))
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                 sampler=sampler, num_workers=num_workers),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
        }
    else:
        dataloaders = {
            'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                 shuffle=True, num_workers=num_workers),
            'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
        }

    return dataloaders, dataset_sizes, class_names
