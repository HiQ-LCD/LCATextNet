import os
import sys 
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .model import MODELS
from .config import DATA_PATH
import os
from .trainer import Trainer
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Dict, List, Union, Tuple
import pandas as pd


class ComplexRegressionErrorAnalyzer:
    """Complex regression model error analyzer"""

    def __init__(self, model: nn.Module):
        self.model = model
    
    def analyze_errors(self,
                       inputs: Tuple[torch.Tensor, ...],
                       true_values: torch.Tensor) -> Dict:
        """Analyze prediction errors"""

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(*inputs).cpu().numpy()
        true_values = true_values.cpu().numpy()

        errors = predictions - true_values

        r2 = r2_score(true_values, predictions)

        error_analysis = {
            'r2': r2,
            'mse': np.mean(errors ** 2),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'error_distribution': errors
        }

        return error_analysis

    def export_prediction_results(self, inputs: Tuple[torch.Tensor, ...], true_values: torch.Tensor,
                                  output_path):
        """Export prediction results"""
        self.model.eval()
        with torch.no_grad():
            # Print data type and device for each input tensor
            for i, input_tensor in enumerate(inputs):
                print(f"Input {i}:")
                print(f"  Type: {type(input_tensor)}")
                print(f"  Dtype: {input_tensor.dtype}")
                print(f"  Device: {input_tensor.device}")
                print(f"  Shape: {input_tensor.shape}")
                print()
            predictions = self.model(*inputs).cpu().numpy()
        true_values = true_values.cpu().numpy()
        errors = predictions - true_values

        df = pd.DataFrame({'prediction': predictions, 'true_value': true_values, 'error': errors})
        df.to_csv(output_path, index=False)

        r2 = r2_score(true_values, predictions)

        error_analysis = {
            'r2': r2,
            'mse': np.mean(errors ** 2),
            'rmse': np.sqrt(np.mean(errors ** 2)),
            'mae': np.mean(np.abs(errors)),
            'error_distribution': errors
        }

        return error_analysis


def evaluate_model(model_name, train_date, model_version,result_path,impact_name=None):
    model_version = f'{train_date}_{model_version}'
    model_folder = os.path.join(DATA_PATH,'gwp', 'model', train_date)
    model_config_path = os.path.join(model_folder, f"{model_name}_{model_version}_config.json")
    model_path = os.path.join(model_folder, f"{model_name}_{model_version}.bin")
    model_config = json.load(open(model_config_path))
    model_config['model_path'] = model_path
    model = MODELS[model_name](**model_config)

    trainer = Trainer(model_name)
    (
        text_inputs_embeddings, system_boundary_embeddings, true_values) = trainer.load_data(impact_name)

    # initialize model and explainer
    explainer = ComplexRegressionErrorAnalyzer(model)

    # prepare input data
    inputs = (text_inputs_embeddings, system_boundary_embeddings)

    # perform analysis
    results = explainer.export_prediction_results(inputs, true_values, result_path)
    print(results)
    return