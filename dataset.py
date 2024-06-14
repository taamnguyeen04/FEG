from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize, ToTensor, Compose


class EmotionImgDataset(Dataset):
    def __init__(self, root, is_train, transform=None):
        if is_train:
            data_path = os.path.join(root, "images")
        else:
            data_path = os.path.join(root, "test")
        self.categories = ["Anger", "Contempt", "Disgust","Fear", "Happy", "Neutral", "Sad", "Surprised"]
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
        img_path = os.path.join(root, "image")
        img_path = os.path.join(img_path, "origin")
        label_path =os.path.join(root, "label")
        label_path =os.path.join(label_path, "label.lst")
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



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = Compose([
        ToTensor(),
        Resize((224, 224))
    ])
    train_dataset = ExpW(root="data/ExpW/data", transform=transform)
    img, label = train_dataset[1]
    print(img.shape, label)



    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        num_workers=8,
        shuffle=True,
        drop_last=True
    )
    print(train_dataloader)
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)