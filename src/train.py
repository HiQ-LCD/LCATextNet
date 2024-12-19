# -*- coding: utf-8 -*-
# @Time    : 2024/12/19 13:31
# @Author  : Biao
# @File    : train.py

import os.path
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from .trainer import Trainer
from .config import MODEL_ROOT_PATH, DATA_PATH
from .evaluate import evaluate_model
import json
from .dataset import ENVIRONMENTAL_IMPACT_DICT

with open("model_config.json", "r") as f:
    config_data = json.load(f)

model_name = config_data["model_name"]
model_params = config_data["model_config"][model_name]
train_config = config_data["train_config"]
batch_size = train_config["batch_size"]
num_epochs = train_config["epochs"]
learning_rate = train_config["learning_rate"]
weight_decay = train_config["weight_decay"]
loss_func = train_config["loss_func"]
model_version = train_config["train_version"]
train_params = {
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "loss_func": loss_func,
}

fig_path = os.path.join(MODEL_ROOT_PATH, "gwp_predict")

model_save_folder = os.path.join(DATA_PATH, "gwp", "model", f"{model_version}")

impact_dict = ENVIRONMENTAL_IMPACT_DICT

valid_impact_index = [1]
impact_dict = {k: v for k, v in impact_dict.items() if k in valid_impact_index}

for idx,impact in impact_dict.items():
    impact_name = impact["impact_name"]
    iqr_mean = impact["iqr_mean"]
    scale_factor = impact["scale_factor"]
    trainer = Trainer(
        model_name=model_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        model_params=model_params, 
    )
    print(f"Training model for {impact_name}")  
    print('iqr_mean:',iqr_mean)
    print('scale_factor:',scale_factor)
    text_inputs_embeddings, system_boundary_embeddings, labels = trainer.load_data(
        impact_name
    )
    print(text_inputs_embeddings.shape, system_boundary_embeddings.shape, labels.shape)
    train_loader, val_loader = trainer.create_dataloaders(
        text_inputs_embeddings, system_boundary_embeddings, labels
    )
    train_losses, val_losses, train_evaluate_metrics, val_evaluate_metrics, model_path = (
        trainer.train(
            train_loader,
            val_loader,
            num_epochs,
            loss_func=loss_func,
            model_save_folder=model_save_folder,
            scale_factor=scale_factor
        )
    )

    trainer.plot_training(
        train_losses,
        val_losses,
        train_evaluate_metrics,
        val_evaluate_metrics,
        model_params,
        train_params,
        model_save_folder,
    )


    for name, param in trainer.model.named_parameters():
        if param.requires_grad:
            print(name, param.grad.abs().mean())


    train_date = model_path.split("_")[-2]
    model_version = model_path.split("_")[-1].split(".")[0]
    print(f"Evaluating model {model_name} with version {model_version}")
    rewrite_impact_name=impact_name
    rewrite_impact_name = rewrite_impact_name.replace(" ", "").replace(":", "").replace("/", "_")
    result_path = os.path.join(
        DATA_PATH,
        "gwp",
        "evaluate_results",
        f"{model_name}_{train_date}_{model_version}_{rewrite_impact_name}.csv",
    )


    evaluate_model(model_name, train_date, model_version, result_path, impact_name)
