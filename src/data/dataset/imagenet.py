import torch
import torchvision.transforms
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize
from functools import partial

def center_crop_fn(image, height, width):
    crop_x = (image.width - width) // 2
    crop_y = (image.height - height) // 2
    return image.crop((crop_x, crop_y, crop_x + width, crop_y + height))


class LocalCachedDataset(ImageFolder):
    def __init__(self, root, resolution=256, cache_root=None):
        super().__init__(root)
        self.cache_root = cache_root
        self.transform = partial(center_crop_fn, height=resolution, width=resolution)

    def load_latent(self, latent_path):
        pk_data = torch.load(latent_path)
        mean = pk_data['mean'].to(torch.float32)
        logvar = pk_data['logvar'].to(torch.float32)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        latent = mean + torch.randn_like(mean) * std
        return latent

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        latent_path = image_path.replace(self.root, self.cache_root) + ".pt"

        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)
        if self.cache_root is not None:
            latent = self.load_latent(latent_path)
        else:
            latent = raw_image

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return latent, target, metadata


class PixImageNet(ImageFolder):
    def __init__(self, root, resolution=256, random_crop=False, ):
        super().__init__(root)
        if random_crop:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(resolution),
                    torchvision.transforms.RandomCrop(resolution),
                    torchvision.transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.transform = partial(center_crop_fn, height=resolution, width=resolution)
        self.normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = self.transform(raw_image)
        raw_image = to_tensor(raw_image)

        normalized_image = self.normalize(raw_image)

        metadata = {
            "raw_image": raw_image,
            "class": target,
        }
        return normalized_image, target, metadata