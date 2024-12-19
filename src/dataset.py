# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 11:36
# @Author  : Biao
# @File    : dataset.py

"""
Training with Ecoinvent 3.10 Database, with the following data preprocessing:
1. Unit filtering to retain only kg/kWh/MJ measurements, reducing dataset from 23,523 to 18,148 samples

Input Features:
1. Activity Name: 768-dimensional text embedding vector
2. Reference Product Name: 768-dimensional text embedding vector  
3. CPC Classification: 768-dimensional text embedding vector
4. Product Information: 768-dimensional text embedding vector
5. SystemBoundary: 768-dimensional text embedding vector derived from concatenation of includedActivitiesStart and includedActivitiesEnd, processed through a classifier
6. generalComment: 768-dimensional text embedding vector
7. technologyComment: 768-dimensional text embedding vector

Output Features:
25 environmental impact categories under EF v3.1 methodology
"""
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from .embedding_model import RemoteEmbeddingModel
from .config import DATA_PATH, logger

embedding_model = RemoteEmbeddingModel()

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List
import joblib
from tqdm import tqdm


ENVIRONMENTAL_IMPACT_LIST = [
    "acidification",
    "climate change",
    "climate change: biogenic",
    "climate change: fossil",
    "climate change: land use and land use change",
    "ecotoxicity: freshwater",
    "ecotoxicity: freshwater, inorganics",
    "ecotoxicity: freshwater, organics",
    "energy resources: non-renewable",
    "eutrophication: freshwater",
    "eutrophication: marine",
    "eutrophication: terrestrial",
    "human toxicity: carcinogenic",
    "human toxicity: carcinogenic, inorganics",
    "human toxicity: carcinogenic, organics",
    "human toxicity: non-carcinogenic",
    "human toxicity: non-carcinogenic, inorganics",
    "human toxicity: non-carcinogenic, organics",
    "ionising radiation: human health",
    "land use",
    "material resources: metals/minerals",
    "ozone depletion",
    "particulate matter formation",
    "photochemical oxidant formation: human health",
    "water use",
]

iqr_means = [
    0.005046043956201201,
    0.9462458810520248,
    0.0010387459701092022,
    0.7299265057929034,
    0.0010098377550103322,
    8.269162710815856,
    4.57313857240513,
    1.1470798270563027,
    9.245706446935403,
    0.0002822698806778314,
    0.0011474361412993266,
    0.01032610447800491,
    3.408022048163542e-09,
    1.4500224960716873e-10,
    2.519877812764961e-09,
    1.2730737462430473e-08,
    1.0858180740117762e-08,
    6.832095335457354e-10,
    0.033662736531146584,
    6.080733629798855,
    3.956199746098691e-06,
    1.2060151251377034e-08,
    4.9819113734841856e-08,
    0.0033153869051948926,
    0.3685201653992303,
]

ENVIRONMENTAL_IMPACT_DICT = {
    0: {
        "impact_name": "acidification",
        "iqr_mean": iqr_means[0],
        "scale_factor": 1 / iqr_means[0],
    },
    1: {
        "impact_name": "climate change",
        "iqr_mean": iqr_means[1],
        "scale_factor": 1 / iqr_means[1],
    },
    2: {
        "impact_name": "climate change: biogenic",
        "iqr_mean": iqr_means[2],
        "scale_factor": 1 / iqr_means[2],
    },
    3: {
        "impact_name": "climate change: fossil",
        "iqr_mean": iqr_means[3],
        "scale_factor": 1 / iqr_means[3],
    },
    4: {
        "impact_name": "climate change: land use and land use change",
        "iqr_mean": iqr_means[4],
        "scale_factor": 1 / iqr_means[4],
    },
    5: {
        "impact_name": "ecotoxicity: freshwater",
        "iqr_mean": iqr_means[5],
        "scale_factor": 1 / iqr_means[5],
    },
    6: {
        "impact_name": "ecotoxicity: freshwater, inorganics",
        "iqr_mean": iqr_means[6],
        "scale_factor": 1 / iqr_means[6],
    },
    7: {
        "impact_name": "ecotoxicity: freshwater, organics",
        "iqr_mean": iqr_means[7],
        "scale_factor": 1 / iqr_means[7],
    },
    8: {
        "impact_name": "energy resources: non-renewable",
        "iqr_mean": iqr_means[8],
        "scale_factor": 1 / iqr_means[8],
    },
    9: {
        "impact_name": "eutrophication: freshwater",
        "iqr_mean": iqr_means[9],
        "scale_factor": 1 / iqr_means[9],
    },
    10: {
        "impact_name": "eutrophication: marine",
        "iqr_mean": iqr_means[10],
        "scale_factor": 1 / iqr_means[10],
    },
    11: {
        "impact_name": "eutrophication: terrestrial",
        "iqr_mean": iqr_means[11],
        "scale_factor": 1 / iqr_means[11],
    },
    12: {
        "impact_name": "human toxicity: carcinogenic",
        "iqr_mean": iqr_means[12],
        "scale_factor": 1 / iqr_means[12],
    },
    13: {
        "impact_name": "human toxicity: carcinogenic, inorganics",
        "iqr_mean": iqr_means[13],
        "scale_factor": 1 / iqr_means[13],
    },
    14: {
        "impact_name": "human toxicity: carcinogenic, organics",
        "iqr_mean": iqr_means[14],
        "scale_factor": 1 / iqr_means[14],
    },
    15: {
        "impact_name": "human toxicity: non-carcinogenic",
        "iqr_mean": iqr_means[15],
        "scale_factor": 1 / iqr_means[15],
    },
    16: {
        "impact_name": "human toxicity: non-carcinogenic, inorganics",
        "iqr_mean": iqr_means[16],
        "scale_factor": 1 / iqr_means[16],
    },
    17: {
        "impact_name": "human toxicity: non-carcinogenic, organics",
        "iqr_mean": iqr_means[17],
        "scale_factor": 1 / iqr_means[17],
    },
    18: {
        "impact_name": "ionising radiation: human health",
        "iqr_mean": iqr_means[18],
        "scale_factor": 1 / iqr_means[18],
    },
    19: {
        "impact_name": "land use",
        "iqr_mean": iqr_means[19],
        "scale_factor": 1 / iqr_means[19],
    },
    20: {
        "impact_name": "material resources: metals/minerals",
        "iqr_mean": iqr_means[20],
        "scale_factor": 1 / iqr_means[20],
    },
    21: {
        "impact_name": "ozone depletion",
        "iqr_mean": iqr_means[21],
        "scale_factor": 1 / iqr_means[21],
    },
    22: {
        "impact_name": "particulate matter formation",
        "iqr_mean": iqr_means[22],
        "scale_factor": 1 / iqr_means[22],
    },
    23: {
        "impact_name": "photochemical oxidant formation: human health",
        "iqr_mean": iqr_means[23],
        "scale_factor": 1 / iqr_means[23],
    },
    24: {
        "impact_name": "water use",
        "iqr_mean": iqr_means[24],
        "scale_factor": 1 / iqr_means[24],
    },
}

SCALE_FACTORS = [1 / iqr_mean for iqr_mean in iqr_means]


class ImpactPredictDataset(Dataset):
    def __init__(
        self,
        text_inputs_embeddings: List,
        system_boundary_embeddings: List,
        impacts_values: List,
    ):
        self.text_inputs_embeddings = text_inputs_embeddings
        self.system_boundary_embeddings = system_boundary_embeddings
        self.impacts_values = impacts_values

        self.labels = self.impacts_values
        VALID_INDICES = joblib.load(
            os.path.join(DATA_PATH, "gwp", "include_indices.pkl")
        )

        np_values = np.array(torch.tensor(self.impacts_values).to(torch.device("cpu")))
        valid_indices = VALID_INDICES
        self.labels = self.labels[valid_indices]
        self.text_inputs_embeddings = self.text_inputs_embeddings[valid_indices]
        self.system_boundary_embeddings = self.system_boundary_embeddings[valid_indices]
        logger.info(f"data length: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text_inputs_embeddings[idx],
            self.system_boundary_embeddings[idx],
            self.labels[idx],
        )


class GWPPredictDataset(Dataset):
    def __init__(
        self,
        text_inputs_embeddings: List,
        system_boundary_embeddings: List,
        gwp_values: List,
    ):
        self.text_inputs_embeddings = text_inputs_embeddings
        self.system_boundary_embeddings = system_boundary_embeddings
        self.gwp_values = gwp_values
        self.labels = self.gwp_values

        logger.info(f"data length: {len(self.labels)}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.text_inputs_embeddings[idx],
            self.system_boundary_embeddings[idx],
            self.labels[idx],
        )
