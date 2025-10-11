# -*- coding: UTF-8 -*-
# @Time    : 2024/06/07  
# @Author  : Gemini
# @File    : loaders_radio_lightm_unet.py
# @Description: Data loader for Radio-MambaNet v14.
#              Same data loading logic as v12, compatible with simplified v14 configuration.

from __future__ import print_function, division
import os
import warnings
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

warnings.filterwarnings("ignore")


class RadioMambaNetDataset(Dataset):
    """
    RadioMapSeer Data Loader for Radio-MambaNet.
    Updated to support both with/without cars simulation and input.
    Input: Building map (1 channel), Transmitter location map (1 channel), and cars/buildings channel (1 channel).
    Output: DPM path loss map (1 channel).
    """

    def __init__(self, maps_inds=None, phase="train",
                 ind1=0, ind2=0,
                 thresh = 0.0,
                 dir_dataset="RadioMapSeer/",
                 numTx=80,
                 carsSimul="no",
                 carsInput="no",
                 transform=transforms.ToTensor()):
        """
        Args:
            maps_inds (np.ndarray, optional): Shuffled map indices.
            phase (str): "train", "val", or "test".
            ind1, ind2 (int): Start and end indices for custom data splits.
            dir_dataset (str): Root directory of the RadioMapSeer dataset.
            numTx (int): Number of transmitters per map.
            carsSimul (str): "no" or "yes". Use simulation with or without cars. Default="no".
            carsInput (str): "no" or "yes". Take inputs with or without cars channel. Default="no".
            transform (callable, optional): Transform to be applied to the images.
        """

        if maps_inds is None:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 549
        elif phase == "val":
            self.ind1 = 550
            self.ind2 = 599
        elif phase == "test":
            self.ind1 = 600
            self.ind2 = 699
        else:  # custom
            self.ind1 = ind1
            self.ind2 = ind2

        self.thresh = thresh
        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.carsSimul = carsSimul
        self.carsInput = carsInput

        # Setup gain directory based on cars simulation
        if carsSimul == "no":
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "DPM")
        else:
            self.dir_gain = os.path.join(self.dir_dataset, "gain", "carsDPM")
        
        self.dir_buildings = os.path.join(
            self.dir_dataset, "png", "buildings_complete")
        self.dir_Tx = os.path.join(self.dir_dataset, "png", "antennas")
        
        # Setup cars directory if needed
        if carsInput != "no":
            self.dir_cars = os.path.join(self.dir_dataset, "png", "cars")

        self.transform = transform
        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):
        map_idx_in_split = idx // self.numTx
        tx_idx_in_map = idx % self.numTx
        dataset_map_ind = self.maps_inds[self.ind1 + map_idx_in_split] + 1

        name1 = str(dataset_map_ind) + ".png"
        name2 = str(dataset_map_ind) + "_" + str(tx_idx_in_map) + ".png"

        # Load building map
        img_name_buildings = os.path.join(self.dir_buildings, name1)
        image_buildings = np.asarray(
            io.imread(img_name_buildings), dtype=np.float32) / 255.0

        # Load transmitter map
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx), dtype=np.float32) / 255.0

        # Load ground truth gain map
        img_name_gain = os.path.join(self.dir_gain, name2)
        image_gain = np.expand_dims(
            io.imread(img_name_gain).astype(np.float32), axis=2) / 255.0
        
        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain = image_gain/(1-self.thresh)

        # Ensure channel dimension exists
        if image_buildings.ndim == 2:
            image_buildings = np.expand_dims(image_buildings, axis=2)
        if image_Tx.ndim == 2:
            image_Tx = np.expand_dims(image_Tx, axis=2)

        # Prepare third channel based on cars input setting
        if self.carsInput == "no":
            # Use buildings as third channel (same as before)
            third_channel = image_buildings
        else:
            # Load cars map for third channel
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars), dtype=np.float32) / 255.0
            if image_cars.ndim == 2:
                image_cars = np.expand_dims(image_cars, axis=2)
            third_channel = image_cars

        # Concatenate to form a 3-channel input
        # (Buildings, Tx, Buildings/Cars)
        inputs_numpy = np.concatenate(
            [image_buildings, image_Tx, third_channel], axis=2)

        if self.transform:
            inputs = self.transform(inputs_numpy).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        else:  # Fallback if no transform is provided
            inputs = torch.from_numpy(
                inputs_numpy.transpose((2, 0, 1))).type(torch.float32)
            image_gain = torch.from_numpy(
                image_gain.transpose((2, 0, 1))).type(torch.float32)

        return inputs, image_gain, name2
