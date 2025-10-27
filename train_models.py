import json
import os
import sys
import torch
sys.path.append('./src/')
from sklearn import metrics
import numpy as np
import math
from src import losses
from src.models import RandomFeatureGaussianProcess
from src.utils import train_model, create_moons_data_loaders
from src.viz_utils import plot_preds, compute_probabilities_gp
import torch

input_size = 2
output_size = 2
n_epochs = 20_000
do_early_stopping = True
show_fig = False

train_loader, val_loader, test_loader = create_moons_data_loaders(3000, 3000)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA:", torch.cuda.get_device_name(device))
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Metal)")
else:
    device = torch.device("cpu")
    print("Using CPU")

print("Final device:", device)

## ALL GRID-SEARCH PARAMETERS 
lengthscale_grid = [0.01, 0.05, 0.1, 0.25, 0.50]
outputscale_grid = [0.1, 0.5, 1.0, 2.0]
learning_rate_grid = [0.5]
ranks = [1_024, 5_000, 10_000]

results_dir = "grid_search_results"
os.makedirs(results_dir, exist_ok=True)

results = []
for lr in learning_rate_grid:
    for lengthscale in lengthscale_grid:
        for outputscale in outputscale_grid:
            for rank in ranks:
                run_label = f"rank{rank}_ls{lengthscale}_lr{lr}"
                print(f"\nStarting training for {run_label}")

                os.makedirs(f'./results/script_results/{run_label}/', exist_ok=True)
                
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
                    model_filename=f'./results/script_results/{run_label}/best_model.pth',
                )

                preds_filename = f"./results/script_results/{run_label}/preds.pt"
                fig_filename = f"./results/script_results/{run_label}/plots.png"

                compute_probabilities_gp(trained_model, train_loader, preds_filename=preds_filename, compute_covariance=True, device=device, num_samples=1000)
                plot_preds(preds_filename, test_loader, fig_filename, show_fig=show_fig)

                final_loss = info['best_va_loss']

                results.append({
                    "outputscale": outputscale,
                    "lengthscale": lengthscale,
                    "learning_rate": lr,
                    "final_loss": final_loss,
                    "pred_file": preds_filename,
                    "fig_filename": fig_filename,
                })

results_file = os.path.join(results_dir, "./results/script_results/grid_search_results.json")
with open(results_file, "w") as f:
    json.dump(results, f, indent=4)

print("Grid search complete. Results saved to:", results_file)