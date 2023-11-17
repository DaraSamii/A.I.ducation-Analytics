import os
from tqdm import tqdm
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid

import PIL.Image as Image

import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models


#################################################################################################################
class ImageDataset(Dataset):
    def _label_func(self, file_path):
        return file_path.split("/")[-2]

    def __init__(self,path,):
        self.path = os.path.abspath(path)
        self.folders = os.listdir(path)

        self.files = []
        for folder in self.folders:
            files = os.listdir(os.path.join(self.path,folder))
            abspath_fils = list(map(lambda x: os.path.join(self.path,folder,x),files))
            self.files.extend(abspath_fils)

        self.labels = list(map(lambda x:self._label_func(x),self.files))

        self.transforms = transforms.Compose([
                                              transforms.Resize(size=(48, 48)),
                                              transforms.ToTensor(),
                                              transforms.RandomHorizontalFlip(p= 0.5),
                                              transforms.RandomRotation(degrees=(-10,10)),

        ])

        self.classes = list(set(self.labels))
        self.n_class = len(self.classes)
        self.classes_dict = dict()
        for idx,cls in enumerate(self.classes):
            self.classes_dict[cls] = idx
    def __len__(self):
        return len(self.files)


    def __getitem__(self, idx):

        Im=self.transforms(Image.open(self.files[idx]).convert(mode='L'))
        label = F.one_hot(T.tensor(self.classes_dict[self.labels[idx]]),self.n_class)

        return Im, label
    
    ######################################################################################################################################

class Learner:
    def __init__(self, train_dl, val_dl, model,labels_name, base_lr=0.001, base_wd = 0.0,save_best=None,log_path=None):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.model = model
        self.labels_name = labels_name
        self.save_best =  save_best
        self.log_path = log_path
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.epoch = 0
        self.base_lr = base_lr
        self.base_wd = base_wd
        self.criterion = nn.CrossEntropyLoss()
        self.optim = optim.Adam(self.model.parameters(), lr=self.base_lr,weight_decay=self.base_wd)

        self.logs = []
        self.metrics = {}
        self.best_score = 0.0

    def train_one_epoch(self,lr,wd=0.0):
        self.model.train()
        self.lr = lr
        self.wd=wd

        for g in self.optim.param_groups:
            g['lr'] = self.lr

        for g in self.optim.param_groups:
            g['weight_decay'] = self.wd

        self.model.to(self.device)
        self.model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(self.train_dl):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optim.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.to(T.float))

            loss.backward()
            self.optim.step()

            train_loss += loss.item()

        self.metrics["train loss"] = train_loss / len(self.train_dl)
        return train_loss

    def eval(self):

        self.model.eval()

        self.val_loss = 0.0
        self.all_predictions = []
        self.all_labels = []

        with T.no_grad():
            for inputs, labels in tqdm(self.val_dl, position=0, leave=True):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.outputs = self.model(inputs)
                loss = self.criterion(self.outputs, labels.to(T.float))
                self.val_loss += loss.item()

                _, predictions = T.max(self.outputs, 1)

                self.all_predictions.extend(predictions.cpu().numpy())
                self.all_labels.extend(np.argmax(labels.cpu().numpy(),axis=1))

        # Compute metrics
        accuracy = accuracy_score(self.all_labels, self.all_predictions)
        precision_micro = precision_score(self.all_labels, self.all_predictions, average='micro',zero_division=0)
        recall_micro = recall_score(self.all_labels, self.all_predictions, average='micro')
        f1_micro = f1_score(self.all_labels, self.all_predictions, average='micro')

        precision_macro = precision_score(self.all_labels, self.all_predictions, average='macro',zero_division=0)
        recall_macro = recall_score(self.all_labels, self.all_predictions, average='macro')
        f1_macro = f1_score(self.all_labels, self.all_predictions, average='macro')
        self.metrics['Validation Loss'] = self.val_loss / len(self.val_dl)

        self.metrics['Accuracy']= accuracy
        self.metrics['Precision Micro']= precision_micro
        self.metrics['Recall Micro']= recall_micro
        self.metrics['F1 Score Micro']= f1_micro
        self.metrics['Precision Macro']= precision_macro
        self.metrics['Recall Macro']= recall_macro
        self.metrics['F1 Score Macro']= f1_macro

    def print_metrics(self):
        formatted_metrics = {key: f"{value:.{4}f}" if isinstance(value, (float, np.float32, np.float64)) else value
                                for key, value in self.metrics.items()}

        metric_str = ", ".join([f"{key}: {value}" for key, value in formatted_metrics.items()])
        print(metric_str)

    def train_eval(self,lr,epochs,wd=0.0):
        for _ in range(epochs):
            self.metrics = {"epoch": self.epoch}
            self.train_one_epoch(lr,wd)
            self.eval()
            self.logs.append(self.metrics)
            self.epoch += 1
            self.print_metrics()

            if self.metrics["Accuracy"] > self.best_score:
                self.best_score = self.metrics["Accuracy"]
                if self.save_best != None:
                    self.save(self.save_best)
                    print("\nNew best score -- model saved")
            self.save_log(self.log_path)

    def save_log(self,path):
        df = pd.DataFrame(self.logs)
        df.to_csv(path + ".csv")

    def save(self,path):
        T.save(self.model.state_dict(), path)
        self.save_log(self.log_path)

    def load(self,path):
        state = T.load(path)
        self.model.load_state_dict(state_dict=state,strict=False)

    def load_log(self,path):
        df = pd.read_csv(path)
        self.log = df.to_json()

    def plot_metrics_micro(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df['epoch'],df['Accuracy'],label="Accuracy")
        plt.plot(df['epoch'],df['Precision Micro'],label="Precision Micro'")
        plt.plot(df['epoch'],df['Recall Micro'],label="Recall Micro")
        plt.plot(df['epoch'],df['F1 Score Micro'],label="F1 Score Micro")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("score")
        if save_path != None:
            plt.savefig(save_path)


    def plot_metrics_macro(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df['epoch'],df['Accuracy'],label="Accuracy")
        plt.plot(df['epoch'],df['Precision Macro'],label="Precision Macro'")
        plt.plot(df['epoch'],df['Recall Macro'],label="Recall Macro")
        plt.plot(df['epoch'],df['F1 Score Macro'],label="F1 Score Macro")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("score")
        if save_path != None:
            plt.savefig(save_path)

            
    def plot_loss(self,save_path=None):
        df = pd.DataFrame(self.logs)
        self.df=df
        plt.figure(figsize=(15,10))
        plt.plot(df["epoch"],df["train loss"], label = "Train Loss")
        plt.plot(df["epoch"],df["Validation Loss"],label = "Validation Loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        if save_path != None:
            plt.savefig(save_path)

    def predict(self, test_dl):
        if T.cuda.is_available():
            self.model.cuda()
            
        all_predictions = []
        all_labels = []
        all_probabilites = []
        self.model.eval()
        with T.no_grad():
            for inputs, labels in tqdm(test_dl, position=0, leave=True):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.outputs = self.model(inputs)

                _, predictions = T.max(self.outputs, 1)
                probabilities = T.nn.functional.softmax(self.outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(np.argmax(labels.cpu().numpy(),axis=1))
                all_probabilites.extend(probabilities.cpu().numpy())

        self.test_preds = all_predictions
        self.test_labels = all_labels
        self.test_pred_prob = all_probabilites

        return (all_predictions,all_labels, all_probabilites)
    

    def plotConfisuionMatrix(self,save_path):
        cm = confusion_matrix(self.test_preds, self.test_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=self.labels_name)
        disp.plot(cmap=plt.cm.hot_r)
        plt.savefig(save_path)
################################################################################################################################################


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = kernel_size//2),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = kernel_size//2),
                        nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,num_classes, num_channel, num_blocks, num_fc_layers, avg_pool_size, kernel_size = 3,):
        super(ResNet, self).__init__()

        self.resblocks = self._make_resblocks(num_channel, num_blocks,kernel_size)

        self.avg_pool = nn.AdaptiveAvgPool2d((avg_pool_size, avg_pool_size))

        self.fc_layers = self._make_fc_layers(num_channel* avg_pool_size * avg_pool_size, num_fc_layers,num_classes)

    def _make_resblocks(self, bloc_size, num_resblocks,kernel_size):
        layers = []
        layers.append(ResBlock(1, bloc_size,kernel_size))
        for _ in range(num_resblocks-1):
            layers.append(ResBlock(bloc_size, bloc_size,kernel_size))
        return nn.Sequential(*layers)

    def _make_fc_layers(self, fc_size, num_fc_layers,num_classes):
        layers = []
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(fc_size, fc_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(fc_size,num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.resblocks(x)
        x = self.avg_pool(x)
        x = T.flatten(x, 1)
        x = self.fc_layers(x)
        return x
