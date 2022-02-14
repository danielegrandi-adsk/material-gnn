import csv
import os
import time

import torch
import numpy as np
from glob import glob
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from itertools import chain
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from collections import defaultdict, Counter
from pathlib import Path


def preprocess(data_paths, n_views, force=False):
    for data_path in data_paths:
        models = defaultdict(list)
        images = glob(f'{data_path}/*/*/*/*/*.png')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(500),
            transforms.Resize(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

        def convert(view):
            try:
                view = Image.open(view)
                view = view.convert('RGB')
                view = transform(view)
            except UnidentifiedImageError as e:
                tqdm.write(f"Error: {e}")
                view = torch.zeros(3, 224, 224)
            return view

        for img in images:
            models[Path(img).parent].append(img)

        for model, views in tqdm(models.items(), desc='Converting views'):
            if not Path(f'{model}/views.npy').exists() or force:
                x = list(map(convert, views))
                x = torch.cat((torch.stack(x), torch.zeros(n_views - len(x), *x[0].shape))).detach().numpy()
                # assert x.size() == 0, f"Error, x.size == 0 {model}"
                np.save(f'{model}/views.npy', x)
            assert os.stat(f'{model}/views.npy').st_size > 1000, \
                f"Error: file saved with small size: {model}/views.npy"


class MultiViewDataSet(Dataset):
    def __init__(self, data_paths, split, model_path=None):
        # self.x = list(chain.from_iterable([glob(f'{data_path}/*/{s}/*/*/*.npy') for s in split]))
        self.x = list(chain.from_iterable([glob(f'{data_path}/*/{s}/*/*/*.npy') for s in split
                                           for data_path in data_paths]))
        self.y = [Path(x).parts[-5] for x in self.x]
        if model_path is None:
            self.classes = sorted(list(set(self.y)))
        else:
            with open(Path(model_path) / "labels.csv", newline='') as f:
                reader = csv.reader(f)
                self.classes = list(reader)[0]
        # print(f"Classes: {self.classes}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.y = list(map(self.class_to_idx.get, self.y))
        self.weights = torch.Tensor([len(self.y) / class_images for class_images in Counter(self.y).values()])

    def __getitem__(self, index):
        try:
            x = torch.from_numpy(np.load(self.x[index]))
        except ValueError as e:
            print(f"VALUE ERROR: {e}: {index}, {self.x[index]}")
            raise
        except RuntimeError as e:
            print(f"RUNTIME ERROR: {e}: {index}, {self.x[index]}")
            raise
        except Exception as e:
            print(f"ERROR: {e}: {index}, {self.x[index]}")
            raise
        y = self.y[index]
        assert len(x) > 0, print("ERROR with loading x as tensor")
        return x, y

    def __len__(self):
        return len(self.x)


if __name__ == '__main__':
    path = 'data'
    n_views = 12
    preprocess(path, n_views)


