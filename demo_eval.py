# Standard Libraries
import os
import csv

# External Libraries
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import glob

import torch
from torchvision import transforms, datasets

from networks.dan import DAN


##################################################################################################################
##################################################################################################################
##################################################################################################################


class customTransform:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        #img= Image.fromarray(img,'RGB')

        faces = self.detect(img)

        if len(faces) == 0:
            return 'null'
        #  single face detection
        x, y, w, h = faces[0]
        img = img.crop((x, y, x + w, y + h))
        #img = img.view(1, 3, 224, 224)

        return img


##################################################################################################################
##################################################################################################################
##################################################################################################################

class Model:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger']#, 'contempt']

        self.model = DAN(num_head=4, num_class=7, pretrained=False)

        # FOR PRETRAINED MODELS #
        checkpoint = torch.load('./checkpoints/rafdb_epoch21_acc0.897_bacc0.8275.pth',
                                map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        #                       #

        # FOR PRUNED MODELS     #
        # self.model.load_state_dict(torch.load('./checkpoints/rafdb_epoch21_acc0.897_bacc0.8275_pruned_0.7.pth',map_location=self.device), strict=True)
        #                       #

        self.model.to(self.device)
        self.model.eval()

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    '''
    def detect(self, img0):
        img = cv2.cvtColor(np.asarray(img0), cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img)

        return faces

    def fer(self, path):
        img0 = Image.open(path).convert('RGB')

        faces = self.detect(img0)

        if len(faces) == 0:
            return 'null'

        #  single face detection
        x, y, w, h = faces[0]

        img = img0.crop((x, y, x + w, y + h))

        img = self.data_transforms(img)
        img = img.view(1, 3, 224, 224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out, _, _ = self.model(img)
            _, pred = torch.max(out, 1)
            index = int(pred)
            label = self.labels[index]

            return label
'''
    def fer(self, val_loader):
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(model.device)
                targets = targets.to(model.device)
                out, feat, heads = self.model(imgs)
                _, pred = torch.max(out, 1) # pred contains the indices of the max value for every (batch_size) tensor
                label = [self.labels[i] for i in pred] # contains the labels for every tensor
                print(label)
            return label


##################################################################################################################
##################################################################################################################
##################################################################################################################

model = Model()
batch_size = 5
img_dir= "/home/zachos/Desktop/AffectNet HQ/AffectNetDataset"

data_transforms_val = transforms.Compose([
    customTransform(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


val_dataset = datasets.ImageFolder(img_dir, transform=data_transforms_val)

#if model.num_class == 7:  # ignore the 8-th class
#    idx = [i for i in range(len(val_dataset)) if val_dataset.imgs[i][1] != 7]
#    val_dataset = data.Subset(val_dataset, idx)

print('Validation set size:', val_dataset.__len__())

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         pin_memory=True)

out = model.fer(val_loader)




