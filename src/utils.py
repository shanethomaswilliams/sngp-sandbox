import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import math
import src.losses
import tqdm # type: ignore
from sklearn.datasets import make_moons # type: ignore
from sklearn import metrics # type: ignore
import torch # type: ignore
from torch.utils.data import TensorDataset, DataLoader # type: ignore
import numpy as np # type: ignore

def train_model(model, device, tr_loader, va_loader, loss_module, optimizer=None,
                n_epochs=10, lr=0.001, l2pen_mag=0.0, data_order_seed=42,
                model_filename='best_model.pth',
                do_early_stopping=True,
                n_epochs_without_va_improve_before_early_stop=15,
                ):
    ''' Train model via stochastic gradient descent.

    Assumes provided model's trainable params already set to initial values.

    Returns
    -------
    best_model : PyTorch model
        Model corresponding to epoch with best validation loss (xent)
        seen at any epoch throughout this training run
    info : dict
        Contains history of this training run, for diagnostics/plotting
    '''
    # Make sure tr_loader shuffling reproducible
    torch.manual_seed(data_order_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(data_order_seed)
    model.to(device)
    
    if optimizer is None:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr)

    # Allocate lists for tracking progress each epoch
    tr_info = {'xent':[], 'err':[], 'loss':[]}
    va_info = {'xent':[], 'err':[], 'loss':[]}
    epochs = []

    # Init vars needed for early stopping
    best_va_loss = float('inf')
    curr_wait = 0 # track epochs we are waiting to early stop

    # Count size of datasets, for adjusting metric values to be per-example
    n_train = float(len(tr_loader.dataset))
    n_batch_tr = float(len(tr_loader))
    n_valid = float(len(va_loader.dataset))

    # Progress bar
    progressbar = tqdm.tqdm(range(n_epochs + 1))
    pbar_info = {}

    # Loop over epochs
    for epoch in progressbar:
        if epoch > 0:
            model.train() # In TRAIN mode
            tr_loss = 0.0  # aggregate total loss
            tr_xent = 0.0  # aggregate cross-entropy
            tr_err = 0     # count mistakes on train set
            pbar_info['batch_done'] = 0
            for bb, (x, y) in enumerate(tr_loader):
                optimizer.zero_grad()
                x_BF = x.to(device)
                y_B = y.to(device)

                logits_BC = model(x_BF)
                loss_xent = loss_module.calc_labeled_loss_for_batch(
                    logits_BC, y_B)
                
                params = flatten_params(model)
                l2_loss = l2pen_mag * torch.sum(params ** 2)
                
                loss = loss_xent + float(l2pen_mag) / n_train * l2_loss
                loss.backward()
                optimizer.step()
    
                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)
    
                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batch_tr
                tr_xent += loss_xent.item() / n_batch_tr
                tr_err += metrics.zero_one_loss(
                    logits_BC.argmax(axis=1).detach().cpu().numpy(),
                    y_B.detach().cpu().numpy(), normalize=False)
            tr_err_rate = tr_err / n_train
        else:
            # First epoch (0) doesn't train, just measures initial perf on val
            tr_loss = np.nan
            tr_xent = np.nan
            tr_err_rate = np.nan

        # Track performance on val set
        with torch.no_grad():
            model.eval() # In EVAL mode
            va_total_loss = 0.0
            va_xent = 0.0
            va_err = 0
            num_seen_valid = 0
            num_batches_valid = 0
            for xva_BF, yva_B in va_loader:
                xva_BF = xva_BF.to(device)
                yva_B = yva_B.to(device)
                logits_BC = model(xva_BF)

                va_loss_xent = loss_module.calc_labeled_loss_for_batch(
                    logits_BC, yva_B
                )  # scalar CE on this batch

                # same L2 penalty on *current model params* (not frozen params)
                params = flatten_params(model)
                l2_loss = l2pen_mag * torch.sum(params ** 2)

                # same total objective structure
                va_loss_total = va_loss_xent + float(l2pen_mag) / n_valid * l2_loss

                # bookkeeping
                batch_size = yva_B.shape[0]
                num_seen_valid += batch_size
                num_batches_valid += 1

                va_total_loss += va_loss_total.item()
                va_xent += va_loss_xent.item()

                va_err += metrics.zero_one_loss(
                    logits_BC.argmax(dim=1).detach().cpu().numpy(),
                    yva_B.detach().cpu().numpy(),
                    normalize=False,
                )
            va_avg_total_loss = va_total_loss / num_batches_valid
            va_avg_xent = va_xent / num_batches_valid
            va_err_rate = va_err / num_seen_valid

        # Update diagnostics and progress bar
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        tr_info['xent'].append(tr_xent)
        tr_info['err'].append(tr_err_rate)     
        va_info['loss'].append(va_avg_total_loss)   
        va_info['xent'].append(va_avg_xent)
        va_info['err'].append(va_err_rate)
        pbar_info.update({
            "tr_xent": tr_xent, "tr_err": tr_err_rate,
            "va_xent": va_xent, "va_err": va_err_rate,
            })
        progressbar.set_postfix(pbar_info)

        # Early stopping logic
        # If loss is dropping, track latest weights as best
        if va_xent < best_va_loss:
            best_epoch = epoch
            best_va_loss = va_xent
            best_tr_err_rate = tr_err_rate
            best_va_err_rate = va_err_rate
            curr_wait = 0
            model = model.cpu()
            torch.save(model.state_dict(), model_filename)
            model.to(device)
        else:
            curr_wait += 1
                
        wait_enough = curr_wait >= n_epochs_without_va_improve_before_early_stop
        if do_early_stopping and wait_enough:
            print("Stopped early.")
            break

    print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
    model.to(device)
    model.load_state_dict(torch.load(model_filename, map_location=torch.device('cpu')))    
    result = { 
        'data_order_seed':data_order_seed,
        'lr':lr, 'n_epochs':n_epochs, 'l2pen_mag':l2pen_mag,
        'tr':tr_info,
        'va':va_info,
        'best_tr_err': best_tr_err_rate,
        'best_va_err': best_va_err_rate,
        'best_va_loss': best_va_loss,
        'best_epoch': best_epoch,
        'epochs': epochs}
    return model, result


class _BoundedSpectralNorm(torch.nn.utils.parametrizations._SpectralNorm):
    """
    _BoundedSpectralNorm extends the _SpectralNorm class from PyTorch and adds a bound.
    
    Reference:
        For the original _SpectralNorm class implementation, see:
        https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/parametrizations.py
    """
    def __init__(self, weight, spec_norm_iteration=1, spec_norm_bound=1.0, dim=0, eps=1e-12):
        super(_BoundedSpectralNorm, self).__init__(weight, spec_norm_iteration, dim=0, eps=1e-12)
        self.spec_norm_bound = spec_norm_bound

    def forward(self, weight):
        if weight.ndim == 1:
            sigma = torch.linalg.vector_norm(weight)
            #return F.normalize(weight, dim=0, eps=self.eps)
            return self.spec_norm_bound * F.normalize(weight, dim=0, eps=self.eps) if self.spec_norm_bound < sigma else weight
        else:
            weight_mat = self._reshape_weight_to_matrix(weight)
            if self.training:
                self._power_method(weight_mat, self.n_power_iterations)
            u = self._u.clone(memory_format=torch.contiguous_format)
            v = self._v.clone(memory_format=torch.contiguous_format)
            sigma = torch.vdot(u, torch.mv(weight_mat, v))
            #return weight / sigma
            return self.spec_norm_bound * weight / sigma if self.spec_norm_bound < sigma else weight
        
def apply_bounded_spectral_norm(model, name='weight', spec_norm_iteration=1, spec_norm_bound=1.0):
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            torch.nn.utils.parametrize.register_parametrization(
                module, name, _BoundedSpectralNorm(getattr(module, name, None), spec_norm_iteration=spec_norm_iteration, spec_norm_bound=spec_norm_bound)
            )

def flatten_params(model, excluded_params=['lengthscale_param', 'outputscale_param', 'sigma_param']):
    return torch.cat([param.view(-1) for name, param in model.named_parameters() if name not in excluded_params])

def create_moons_data_loaders(n_samples, batch_size, noise=0.1, random_state=42,
                              train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, shuffle=True):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, val_loader, test_loader

def get_data_from_loader(loader):
    xs, ys = [], []
    for x_batch, y_batch in loader:
        xs.append(x_batch)
        ys.append(y_batch)
    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)
    return x_all.numpy(), y_all.numpy()

def compute_covariance(sngp_model, loader, device='cpu'):
    sngp_model.to(device)
    sngp_model.eval()
    rank = sngp_model.rank
    num_classes = sngp_model.out_features
    precision = torch.stack([torch.eye(rank, device=device) for _ in range(num_classes)])
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            phi = sngp_model.featurize(x)
            logits = sngp_model(x)
            probs = F.softmax(logits, dim=1)
            weights = probs * (1 - probs)
            for k in range(num_classes):
                weights_phi = weights[:, k].unsqueeze(1) * phi
                prec_update_k = weights_phi.T @ phi
                precision[k] += prec_update_k
    ## UNCERTAIN OF THIS BEING DIAG DOM, DIG THROUGH EVIDENCE FOR EXISTING INVERSE
    covariance = torch.stack([torch.inverse(precision[k]) for k in range(num_classes)], dim=0)
    return covariance

    

    
