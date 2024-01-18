import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import Resize
from torchvision.transforms.functional import rgb_to_grayscale
import torch


class CustomImageDataset(Dataset):
    def __init__(self, dataset_path, mode, transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.classes = os.listdir(os.path.join(dataset_path, "train"))
        # self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        def __structure__(self):
            file_list = []
            for class_name in os.listdir(os.path.join(self.dataset_path, self.mode)):
                for filename in os.listdir(
                    os.path.join(self.dataset_path, self.mode, class_name)
                ):
                    file_list.append(
                        os.path.join(self.dataset_path, self.mode, class_name, filename)
                    )
            return file_list

        self.file_path_list = __structure__(self)

        def __transformation__(self, image):
            if transform == "scale_224":
                image = image[:3, :, :]
                # image = rgb_to_grayscale(image)
                resize_transform = Resize((224, 224))
                image = resize_transform(image)
            if transform == "grayscale":
                image = image[:3, :, :]
                image = rgb_to_grayscale(image)

            image = torch.tensor(
                image, dtype=torch.float32, device=torch.device("cuda")
            )
            return image

        self.transform = __transformation__

        def __target_transformation__(self, label):
            label_idx = self.classes.index(label)
            label = torch.zeros(len(self.classes), device=torch.device("cuda"))
            label[label_idx] = 1
            return label

        self.target_transform = __target_transformation__

    def __len__(self):
        count = 0
        for class_name in os.listdir(os.path.join(self.dataset_path, self.mode)):
            count += len(
                os.listdir(os.path.join(self.dataset_path, self.mode, class_name))
            )
        return count

    def __getitem__(self, idx):
        img_path = self.file_path_list[idx]
        image = read_image(img_path)
        label = img_path.split("\\")[2]
        if self.transform:
            image = self.transform(self, image=image)
        if self.target_transform:
            label = self.target_transform(self, label=label)
        return image, label

    # torch.tensor(
    #         self.classes.index(label), device=torch.device("cuda")
    #     )
