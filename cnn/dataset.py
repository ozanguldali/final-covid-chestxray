from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets, transforms


def set_dataset(folder, size=112, normalize=None):
    transform = set_transform(resize=size, crop=size, normalize=normalize)
    dataset = datasets.ImageFolder(folder, transform=transform)

    return dataset


def set_loader(dataset, batch_size=1, shuffle=False, num_workers=1):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def set_transform(resize=112, crop=112, normalize=None, additional=None):
    if normalize is None or normalize is True:
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    transform_list = [
        transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.Grayscale(3)
    ]

    if additional is not None:
        transform_list.extend(additional)

    transform_list.extend([
        transforms.ToTensor()
    ])

    if normalize is None or normalize is True:
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        transform_list.extend([transforms.Normalize(mean=normalize[0], std=normalize[1])])
    elif normalize is not False:
        transform_list.extend([transforms.Normalize(mean=normalize[0], std=normalize[1])])
    else:
        pass

    return transforms.Compose(transform_list)


def normalize_tensor(tensor, norm_value=None):
    if norm_value is None:
        norm_value = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        transform = transforms.Normalize(mean=norm_value[0], std=norm_value[1])
    else:
        transform = transforms.Normalize(mean=norm_value[0], std=norm_value[1])

    return transform(tensor)


def inv_normalize_tensor(tensor, normalize=None):
    if normalize is None or normalize is True:
        normalize = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    mean, std = normalize[0], normalize[1]

    return transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])(tensor)
