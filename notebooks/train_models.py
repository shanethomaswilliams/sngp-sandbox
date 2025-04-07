import json
import os
import sys
import torch
sys.path.append('../src/')
from sklearn import metrics
import numpy as np
import math
import losses
from models import RandomFeatureGaussianProcess
from utils import train_model, create_moons_data_loaders
from viz_utils import plot_preds, compute_probabilities_gp
import torch

input_size = 2
output_size = 2
outputscale = 1.0
n_epochs = 20_000
do_early_stopping = True

train_loader, val_loader, test_loader = create_moons_data_loaders(3000, 3000)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

## ALL GRID-SEARCH PARAMETERS 
lengthscale_grid = [0.01, 0.05, 0.1, 0.25, 0.50]
learning_rate_grid = [2.0, 0.7, 0.5, 0.1, 0.01]
ranks = [1_024, 5_000, 10_000]

results_dir = "grid_search_results"
os.makedirs(results_dir, exist_ok=True)

results = []
for lr in learning_rate_grid:
    for lengthscale in lengthscale_grid:
        for rank in ranks:
            run_label = f"rank{rank}_ls{lengthscale}_lr{lr}"
            print(f"\nStarting training for {run_label}")
            
            # Initialize the model with current hyperparameters
            sngp_model = RandomFeatureGaussianProcess(
                in_features=input_size,
                out_features=output_size,
                rank=rank,
                lengthscale=lengthscale,
                outputscale=1.0
            )

            # Train the model
            trained_model, info = train_model(
                sngp_model,
                device,
                train_loader,
                val_loader,
                losses,
                n_epochs=n_epochs,
                lr=lr,
                do_early_stopping=do_early_stopping,
                model_filename=f'/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/models/model_{run_label}.pth',
            )

            preds_filename = f"/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/preds/preds_{run_label}.pt"
            fig_filename = f"/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/plots/plots_{run_label}.png"

            compute_probabilities_gp(trained_model, train_loader, preds_filename=preds_filename, compute_covariance=True, device=device, num_samples=1000)
            plot_preds(preds_filename, test_loader, fig_filename)

            final_loss = info['best_va_loss']

            results.append({
                "outputscale": outputscale,
                "lengthscale": lengthscale,
                "learning_rate": lr,
                "final_loss": final_loss,
                "pred_file": preds_filename,
                "fig_filename": fig_filename,
            })

results_file = os.path.join(results_dir, "/cluster/tufts/hugheslab/swilli26/SNGP-IMPLEMENTATION/results/grid_search_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print("Grid search complete. Results saved to:", results_file)