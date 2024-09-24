import os
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import pandas as pd

class EmotionImgDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "images")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["angry", "disgust", "fear","happy", "sad", "surprise", "neutral"]
        self.image_paths = []
        self.labels = []
        for dir in os.listdir(data_path):
            list_dir = os.path.join(data_path, dir)
            for category in self.categories:
                self.image_paths.append(os.path.join(list_dir, str(category) + ".jpg"))
                self.labels.append(category)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = cv2.imread(self.image_paths[item])
        # image = Image.open(self.image_paths[item]).convert("RGB")
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label

class ExpW(Dataset):
    def __init__(self, root, transform=None):
        img_path = os.path.join(root, "origin")
        label_path =os.path.join(root, "label.lst")
        self.categories = ["angry", "disgust", "fear","happy", "sad", "surprise", "neutral"]
        self.image_paths = []
        self.labels = []
        self.transform = transform
        with open(label_path, 'r') as file:
            content = file.readlines()
        for line in content:
            line = line.split()
            self.labels.append(int(line[-1]))
            self.image_paths.append(os.path.join(img_path, line[0]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # image = cv2.imread(self.image_paths[item])
        image = Image.open(self.image_paths[item]).convert("RGB")
        # image.show()
        label = self.labels[item]
        if self.transform:
            image = self.transform(image)
        return image, label


class Affectnet(Dataset):
    def __init__(self, root, is_train, transform=None):
        image_path = os.path.join(root, "Manually_Annotated_Images")
        self.transform = transform

        if is_train:
            label_path = os.path.join(root, "training.csv")
        else:
            label_path = os.path.join(root, "validation.csv")

        list_label = pd.read_csv(label_path)
        valid_labels = list_label[list_label['expression'] < 9]

        valid_labels['full_image_path'] = valid_labels['subDirectory_filePath'].apply(
            lambda x: os.path.join(image_path, x))

        valid_labels = valid_labels[valid_labels['full_image_path'].apply(os.path.isfile)]

        self.list_image = valid_labels['full_image_path'].tolist()
        self.list_label_expression = valid_labels['expression'].tolist()

    def __len__(self):
        return len(self.list_label_expression)

    def __getitem__(self, item):
        try:
            image = Image.open(self.list_image[item]).convert("RGB")
            label = self.list_label_expression[item]

            if self.transform:
                image = self.transform(image)

            return image, label

        except (FileNotFoundError, OSError) as e:
            print(f"Error loading image at {self.list_image[item]}: {e}")
            next_item = (item + 1) % len(self.list_image)
            return self.__getitem__(next_item)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])

    train_dataset = Affectnet(root="/home/tam/Desktop/pythonProject1/data/AffectNet", is_train=True, transform=transform)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        print(f"Batch size: {len(images)}, Labels: {labels}")