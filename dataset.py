import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm


class dog_vs_cat_Dataset(Dataset):

    def __init__(self,
                 dataset_path='./data/dog_vs_cat',
                 phase='train',
                 resize=256,
                 cropsize=256,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):
        self.dataset_path = dataset_path
        self.phase = phase
        self.transform = T.Compose([
            T.Resize(resize, Image.Resampling.LANCZOS),
            T.CenterCrop(cropsize),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        self.class_names = []
        self.imgs, self.labels = self.load_dataset_folder()

    def __getitem__(self, idx):

        return self.imgs[idx], self.labels[idx]

    def __len__(self):
        return len(self.imgs)

    def load_dataset_folder(self):
        img_lists = os.listdir(os.path.join(self.dataset_path, self.phase))

        imgs = []
        labels = []

        for img_name in tqdm(img_lists,
                             desc='loading imgs...',
                             total=len(img_lists)):
            img_path = os.path.join(
                os.path.join(self.dataset_path, self.phase), img_name)
            img = self.transform(Image.open(img_path).convert('RGB'))
            imgs.append(img)

            label_name = img_name.split('.')[0]
            if label_name not in self.class_names:
                self.class_names.append(label_name)

            labels.append(np.array([self.class_names.index(label_name)]))

        return list(imgs), list(labels)


if __name__ == '__main__':
    data = dog_vs_cat_Dataset()

    for img, label in data:
        print(img.shape, label.shape)
        break
