from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor, Normalize, CenterCrop, \
    InterpolationMode


class DatasetConfig:
    # Mapping from dataset_id to (train_transforms, test_transforms)
    TRANSFORMS = {
        "uoft-cs/cifar10": (
            Compose([
                # Resize(256, interpolation=InterpolationMode.BILINEAR),
                # CenterCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2470, 0.2435, 0.2616])
            ]),
            Compose([
                # Resize(256, interpolation=InterpolationMode.BILINEAR),
                # CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2470, 0.2435, 0.2616])
            ])
        ),

        # Add other datasets here...
    }

    BATCH_KEY = {
        "uoft-cs/cifar10": "img",
        "flwrlabs/shakespeare": "x",
        "speech_commands": "data"
    }

    BATCH_VALUE = {
        "uoft-cs/cifar10": "label",
        "flwrlabs/shakespeare": "y",
        "speech_commands": "targets"
    }

    @staticmethod
    def get_transform(dataset_id: str, is_train: bool):
        if dataset_id not in DatasetConfig.TRANSFORMS:
            raise ValueError(f"Unknown dataset_id: {dataset_id}")

        transform = DatasetConfig.TRANSFORMS[dataset_id][0 if is_train else 1]
        batch_key = DatasetConfig.BATCH_KEY[dataset_id]

        def apply_transforms(batch):
            batch[batch_key] = [transform(img.convert("RGB")) if isinstance(img, Image.Image) else transform(img) for
                                img in batch[batch_key]]
            return batch

        return apply_transforms
