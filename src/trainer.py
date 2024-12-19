# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 13:32
# @Author  : Biao
# @File    : trainer.py

from .model import MODELS
from .loss import LOSS_FUNCS
from .dataset import GWPPredictDataset,ENVIRONMENTAL_IMPACT_LIST
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import numpy as np
import pandas as pd
from .config import logger, LOG_PATH,MODEL_ROOT_PATH, DATA_PATH
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import json
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import joblib



class Trainer:
    def __init__(self, model_name, batch_size=32, learning_rate=1e-5, weight_decay=0.01,
                 model_params=None):
        if model_params is None:
            model_params = {}
        self.model_params=model_params
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model_name = model_name
        print(self.model_name)
        self.model = MODELS[model_name](**self.model_params).to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logger = logger
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def load_data(self,impact_name):
        activity_name_embeddings = np.array(
            joblib.load(os.path.join(MODEL_ROOT_PATH, "Activity Name_embedding.joblib")))
        logger.info(f"Load activity name embedding shape: {activity_name_embeddings.shape}")
        reference_product_name_embeddings = np.array(joblib.load(
            os.path.join(MODEL_ROOT_PATH, "Reference Product Name_embedding.joblib")))
        logger.info(f"Load reference product name embedding shape: {reference_product_name_embeddings.shape}")
        cpc_classification_embeddings = np.array(joblib.load(
            os.path.join(MODEL_ROOT_PATH, "CPC Classification_embedding.joblib")))
        logger.info(f"Load CPC classification embedding shape: {cpc_classification_embeddings.shape}")
        product_information_embeddings = np.array(joblib.load(
            os.path.join(MODEL_ROOT_PATH, "Product Information_embedding.joblib")))
        logger.info(f"Load product information embedding shape: {product_information_embeddings.shape}")
        general_comment_embeddings = np.array(
            joblib.load(os.path.join(MODEL_ROOT_PATH, "generalComment_embedding.joblib")))
        logger.info(f"Load general comment embedding shape: {general_comment_embeddings.shape}")
        technology_comment_embeddings = np.array(joblib.load(
            os.path.join(MODEL_ROOT_PATH, "technologyComment_embedding.joblib")))
        logger.info(f"Load technology comment embedding shape: {technology_comment_embeddings.shape}")
        system_boundary_embeddings = np.array(
            joblib.load(os.path.join(MODEL_ROOT_PATH, "SystemBoundary_embedding.joblib")))
        system_boundary_embeddings = torch.from_numpy(system_boundary_embeddings).to(torch.float32).to(self.device)

        logger.info(f"Load system boundary embedding shape: {system_boundary_embeddings.shape}")
        if impact_name is not None:
            assert impact_name in ENVIRONMENTAL_IMPACT_LIST
            labels = np.array(
                pd.read_excel(os.path.join(DATA_PATH, "ecoinvent", "gwp_train_data.xlsx"))[impact_name].values.tolist())
        else:
            labels = np.array(
                pd.read_excel(os.path.join(DATA_PATH, "ecoinvent", "gwp_train_data.xlsx"))[ENVIRONMENTAL_IMPACT_LIST].values.tolist())
        labels = torch.from_numpy(labels).to(torch.float32)
        logger.info(f"Load labels shape: {labels.shape}")

        labels = labels.to(self.device)

        activity_name_embeddings = np.expand_dims(activity_name_embeddings, axis=1)
        reference_product_name_embeddings = np.expand_dims(reference_product_name_embeddings, axis=1)
        cpc_classification_embeddings = np.expand_dims(cpc_classification_embeddings, axis=1)
        product_information_embeddings = np.expand_dims(product_information_embeddings, axis=1)
        general_comment_embeddings = np.expand_dims(general_comment_embeddings, axis=1)
        technology_comment_embeddings = np.expand_dims(technology_comment_embeddings, axis=1)
        text_inputs_embeddings = np.concatenate(
            (activity_name_embeddings, reference_product_name_embeddings, cpc_classification_embeddings,
             product_information_embeddings, general_comment_embeddings, technology_comment_embeddings), axis=1)
        text_inputs_embeddings = torch.from_numpy(text_inputs_embeddings).to(torch.float32).to(self.device)
        
        logger.info(f"Concat text inputs embedding shape: {text_inputs_embeddings.shape}")

        return (text_inputs_embeddings, system_boundary_embeddings, labels)

    def create_dataloaders(self, text_inputs_embeddings, system_boundary_embeddings, labels):
        text_inputs_embeddings_train, text_inputs_embeddings_val, system_boundary_embeddings_train, system_boundary_embeddings_val, labels_train, labels_val = train_test_split(
            text_inputs_embeddings, system_boundary_embeddings, labels, test_size=0.1, shuffle=True,
            random_state=30)

        train_dataset = GWPPredictDataset(text_inputs_embeddings_train, system_boundary_embeddings_train,
                                          labels_train)
        val_dataset = GWPPredictDataset(text_inputs_embeddings_val, system_boundary_embeddings_val,
                                        labels_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, train_loader, val_loader, num_epochs=10, loss_func="MSE",model_save_folder=None,scale_factor=1.0):
        self.logger.add(os.path.join(LOG_PATH, "model_train.log"), rotation="10 MB", encoding="utf-8")
        self.logger.info(f"Model {self.model_name} init")
        self.logger.info(f"""Train Params:
        batch_size: {self.batch_size},
        learning_rate: {self.learning_rate},
        weight_decay: {self.weight_decay},""")

        optimizer = Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999))
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        criterion = LOSS_FUNCS[loss_func]()

        train_losses, val_losses, train_evaluate_metrics, val_evaluate_metrics = [], [], [], []
        best_val_loss = float('inf')
        best_model_wts = None

        self.model.apply(self.initialize_weights)

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            preds, targets = [], []
            running_loss = 0.0
            for batch in train_loader:
                text_inputs_embeddings, system_boundary_embeddings, labels = [x.float().to(self.device)
                                                                                            for x in
                                                                                            batch]
                labels = labels * scale_factor
                # print(text_inputs_embeddings.shape, system_boundary_embeddings.shape, labels.shape)
                optimizer.zero_grad()
                outputs = self.model(text_inputs_embeddings, system_boundary_embeddings)
                outputs = outputs * scale_factor
                # print(outputs.shape, labels.shape)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                preds.append(outputs.detach().cpu().numpy())
                targets.append(labels.detach().cpu().numpy())

                running_loss += loss.item() * text_inputs_embeddings.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)
            train_losses.append(epoch_loss)
            train_evaluate_metric= evaluate(preds, targets)
            train_evaluate_metrics.append(train_evaluate_metric)

            # Validation
            self.model.eval()
            preds, targets = [], []
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    text_inputs_embeddings, system_boundary_embeddings, labels = [x.float().to(self.device)
                                                                                            for x in
                                                                                            batch]
                    labels = labels * scale_factor
                    outputs = self.model(text_inputs_embeddings, system_boundary_embeddings)
                    outputs = outputs * scale_factor
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * text_inputs_embeddings.size(0)
                    preds.append(outputs.detach().cpu().numpy())
                    targets.append(labels.detach().cpu().numpy())

            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_losses.append(val_epoch_loss)
            val_evaluate_metric = evaluate(preds, targets)
            val_evaluate_metrics.append(val_evaluate_metric)

            self.logger.info(
                f'Epoch {epoch + 1}/{num_epochs}. Train Loss: {epoch_loss:.4f}, Train R2: {train_evaluate_metric:.4f}, Val Loss: {val_epoch_loss:.4f}, Val R2: {val_evaluate_metric:.4f}. LR = {scheduler.get_last_lr()}')

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_wts = self.model.state_dict().copy()

            scheduler.step(val_epoch_loss)
        if model_save_folder:
            model_path = self.save_model(model_save_folder)
        else:
            model_path = self.save_model(f"{self.model_name}_{self.current_time}.bin")
        self.model.load_state_dict(best_model_wts)
        return train_losses, val_losses, train_evaluate_metrics, val_evaluate_metrics,model_path

    def save_model(self, save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        # 保存模型权重
        model_path = os.path.join(save_folder, f"{self.model_name}_{self.current_time}.bin")
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model {self.model_name} saved to {model_path}")
        
        # 保存模型配置
        model_config_path = os.path.join(save_folder, f"{self.model_name}_{self.current_time}_config.json")
        with open(model_config_path, "w") as f:
            json.dump(self.model.model_config, f)
        logger.info(f"Model config saved to {model_config_path}")
        
        return model_path
    

    def load_scalers(self, scaler_path):
        """
        加载已保存的标准化器
        Args:
            scaler_path: 标准化器文件路径
        """
        self.label_scalers = joblib.load(scaler_path)
        logger.info(f"Label scalers loaded from {scaler_path}")


    def plot_training(self, train_losses, val_losses, train_evaluate_metrics, val_evaluate_metrics, model_params,
                      training_params, path=None):
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Create secondary axis that shares x-axis
        ax2 = ax1.twinx()
        
        # Plot Loss curves on primary axis (left)
        line1 = ax1.plot(train_losses, label='Training Loss', linestyle='-', color='#003f5c')
        line2 = ax1.plot(val_losses, label='Validation Loss', linestyle='--', color='#bc5090')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12, color='#003f5c')
        ax1.tick_params(axis='y', labelcolor='#003f5c')
        
        # Plot R2 curves on secondary axis (right)
        line3 = ax2.plot(train_evaluate_metrics, label='Training R2', linestyle='-', color='#ffa600')
        line4 = ax2.plot(val_evaluate_metrics, label='Validation R2', linestyle='--', color='#ff6361')
        ax2.set_ylabel('R2 Score', fontsize=12, color='#ffa600')
        ax2.tick_params(axis='y', labelcolor='#ffa600')
        
        # Add title
        plt.title('Training Progress', fontsize=14, pad=20)
        
        # Add grid (only for primary axis to avoid cluttering)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Combine legends from both axes
        lines = line1 + line2 + line3 + line4
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
        
        # Add parameters text below the plot
        model_params_str = json.dumps(model_params, separators=(',', ':'))
        training_params_str = json.dumps(training_params, separators=(',', ':'))
        params_text = f"Model Params: {model_params_str}\nTraining Params: {training_params_str}"
        plt.figtext(0.1, 0.01, params_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
        
        # Adjust layout
        plt.tight_layout()
        
        # Create filename and save
        current_time = self.current_time
        filename = f"loss_curve_{current_time}.png"
        
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, filename)
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        
        print(f"Loss curve saved to {filename}")

def evaluate(preds, targets):
    flat_preds = [item for sublist in preds for item in sublist]
    flat_targets = [item for sublist in targets for item in sublist]

    # 计算 R2 分数
    score = r2_score(flat_targets, flat_preds)
    return score