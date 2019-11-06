import numpy as np
import os
from random import choice
import matplotlib.pyplot as plt

class CIFAR10:
    def __init__(self, root, split="l_train", labeled=True):
        self.dataset = np.load(os.path.join(root, "cifar10", split+".npy"), allow_pickle=True).item()
        self.labeled = labeled
        self.equiv_classes = {}

        for i,l in enumerate(self.dataset["labels"]):
            if l in self.equiv_classes:
                self.equiv_classes[l].append(i)
            else:
                self.equiv_classes[l] = [i]

    def __getitem__(self, idx):
        image = self.dataset["images"][idx]
        label = self.dataset["labels"][idx]
        equiv_image = self.dataset["images"][choice([i for i in self.equiv_classes[label] if i != idx])]
        if not self.labeled:
            label = -1
        return image, label, equiv_image

    def __len__(self):
        return len(self.dataset["images"])