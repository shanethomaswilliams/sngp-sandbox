import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.nn.utils as nn_utils
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
from scipy.stats import norm

class StandardThreeLayerDNN(nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(StandardThreeLayerDNN, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)
        
class SingleLayerNetwork(torch.nn.Module):
    def __init__(self, in_features, hidden_size, out_features):
        super(StandardThreeLayerDNN, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return x
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)

class RandomFeatureGaussianProcess(torch.nn.Module):
    def __init__(self, in_features, out_features, learnable_lengthscale=False, learnable_outputscale=False, 
                 lengthscale=0.1, outputscale=1.0, rank=1024):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.learnable_lengthscale = learnable_lengthscale
        self.learnable_outputscale = learnable_outputscale
        if self.learnable_lengthscale:
            self.lengthscale_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(lengthscale))))
        else:
            self.lengthscale_param = torch.log(torch.expm1(torch.tensor(lengthscale)))
        if self.learnable_outputscale:
            self.outputscale_param = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(outputscale))))
        else:
            self.outputscale_param = torch.log(torch.expm1(torch.tensor(outputscale)))
        self.rank = rank
        self.precision_mat = torch.eye(self.rank, device='cpu').unsqueeze(0).repeat(self.out_features, 1, 1)
        self.register_buffer('feature_weight', torch.randn(self.rank, self.in_features))
        self.register_buffer('feature_bias', 2 * torch.pi * torch.rand(self.rank))
        self.linear = torch.nn.Linear(in_features=self.rank, out_features=self.out_features, bias=False)

    def featurize(self, h):
        features = torch.nn.functional.linear(h, (1/self.lengthscale) * self.feature_weight, self.feature_bias)
        return self.outputscale * (2/self.rank)**0.5 * torch.cos(features)
        
    def forward(self, h):
        features = self.featurize(h)
        logits = self.linear(features)
        return logits
    
    def reinitialize_precision(self, device='cpu'):
        precision_CRR = torch.eye(self.rank, device=device).unsqueeze(0).repeat(self.out_features, 1, 1)
        self.precision_mat = precision_CRR
    

    def compute_batch_precision_update(self, h_batch):
        device = self.precision_mat.device

        # Make sure we're on same device
        h_batch = h_batch.to(device)

        ## CALCULATE FEATURES (DIMENSIONS)
        phi_NR = self.featurize(h_batch)         # shape (N, rank)

        ## CALCULATE SOFTMAX PROBABILITIES FROM FEATURES
        logits_NC = self.linear(phi_NR)          # (N, C)
        probs_NC = F.softmax(logits_NC, dim=1)   # (N, C)

        ## FETCH CURRENT PRECISION MATRIX
        cov_inv_CRR = self.precision_mat        # (C, rank, rank)
        N = phi_NR.shape[0]

        for i in range(N):
            ## RE-ORIENT DIMENSIONS OF FEATURES TO CREATE PHI FROM SNGP PAPER
            phi_i_R1 = phi_NR[i].unsqueeze(1)    # (rank, 1)

            ## CREATE OUTER CALCULATION
            outer_RR = phi_i_R1 @ phi_i_R1.t()   # (rank, rank)

            ## CREATE INNER CALCULATION
            var_C = probs_NC[i] * (1.0 - probs_NC[i])  # (C,)

            ## MULTIPLY THE TWO TOGETHER
            cov_inv_CRR = cov_inv_CRR + var_C.view(self.out_features, 1, 1) * outer_RR.unsqueeze(0)

        self.precision_mat = cov_inv_CRR
        
    
    def invert_covariance(self, device='cpu'):
        ## TORCH INVERSE APPROPRIATELY CxRankxRank
        
        ## ENSURING INVERTIBILITY
        ## for i in range(self.out_features):
        ##     covariance_CRR[i] += covariance_CRR[i] + 1e-5 * torch.eye(covariance_CRR.shape[1])
        covariance_CRR = torch.inverse(self.precision_mat)
        return covariance_CRR


    def predict_proba(self, h, covariance, num_samples=100):
        # CALCULAING FEATURES
        features_NR = self.featurize(h)

        ## AGAIN, IN OUR CASE THIS IS EQUIVALENT TO THE ENTIRE TRAINING DATASET
        batch_size = features_NR.shape[0]
        
        ## CALCULATE LOGITS
        mean_logits_NC = self.linear(features_NR)
        
        ## CREATING OUTLINE FOR VARIANCE TO DO MONTE CARLO SAMPLING
        probas_list = []
        for i in range(batch_size):
            logit_C1 = mean_logits_NC[i]
            var_C1 = []
            feature_R1 = features_NR[i]

            ## ITERATING OVER EACH CLASS
            for k in range(self.out_features):
                ## GET COVARIANCE MATRIX FOR SPECIFIC CLASS
                class_cov_RR = covariance[k]
                
                ## CALCULATE VARIANCE FOR SPECIFIC CLASS
                var_k_11 = feature_R1.t() @ class_cov_RR @ feature_R1
                var_C1.append(var_k_11)

            ## CREATE NORMAL DISTRIBUTION FOR ALL CLASSES - https://pytorch.org/docs/stable/distributions.html#normal
            norm = torch.distributions.normal.Normal(loc=logit_C1, scale=torch.sqrt(torch.stack(var_C1, dim=0)))

            ## 2) COMPUTE P(Y|X) BY AVERAGING OVER THE SAMPLES OF M
            samples_BC = norm.sample(sample_shape=(num_samples,))

            ## COMPUTE SOFTMAX
            probs_BC = torch.nn.functional.softmax(samples_BC, dim=-1)

            ## COMPUTE MEAN OF ALL MONTE-CARLO SAMPLES
            model_b_result_C1 = torch.mean(probs_BC, dim=0)
            probas_list.append(model_b_result_C1)
        return torch.stack(probas_list, dim=0)
    
    def update_precision_from_loader(self, train_loader, device='cpu'):
        self.eval()
        cov_inv_CRR = self.precision_mat.to(device)
        with torch.no_grad():
            ## GET ALL DATA FROM TRAIN LOADER
            for X, _ in train_loader:
                X = X.to(device)

                ## CALCULATE PROBABILITIES
                features_NR = self.featurize(X)
                logits_NC = self.linear(features_NR)
                probs_NC = F.softmax(logits_NC, dim=1)
                batch_size = features_NR.shape[0]
                
                for i in range(batch_size):
                    ## RE-ORIENT DIMENSIONS OF FEATURES TO CREATE PHI FROM SNGP PAPER
                    phi_R1 = features_NR[i].unsqueeze(1)
                    outer_product_RR = phi_R1 @ phi_R1.t()

                    ## CREATE INNER CALCULATION
                    inner_calc_C1 = probs_NC[i] * (1 - probs_NC[i])
                    cov_inv_CRR += inner_calc_C1.view(self.out_features, 1, 1) * outer_product_RR.unsqueeze(0)

        ## UPDATE COVARIANCE VARIABLE
        self.precision_mat = cov_inv_CRR
            
    
    @property
    def lengthscale(self):
        return torch.nn.functional.softplus(self.lengthscale_param)
    
    @property
    def outputscale(self):
        return torch.nn.functional.softplus(self.outputscale_param)
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_rate=0.01):
        super().__init__()
        fc = nn.Linear(hidden_size, hidden_size)
        self.fc = fc
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.fc(x)
        out = self.dropout(out)
        out = self.relu(out)
        return x + out
    

class ResFFN12_128(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.01):
        super().__init__()
        hidden_size = 128
        fc_in = nn.Linear(in_features, hidden_size)
        self.fc_in = fc_in
        
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_rate) 
            for _ in range(12)
        ])
        self.fc_out = nn.Linear(hidden_size, out_features)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        logits = self.fc_out(x)
        return logits
    
    def predict_proba(self, x):
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    

class SNGP_ResFFN12_128(nn.Module):
    def __init__(self, in_features, out_features, lengthscale=0.25, outputscale=1.0, rank=1_024, dropout_rate=0.01):
        super().__init__()
        hidden_size = 128
        
        fc_in = nn.Linear(in_features, hidden_size)
        self.fc_in = fc_in
        
        self.relu = nn.ReLU()
    
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(hidden_size, dropout_rate) 
            for _ in range(12)
        ])
        
        self.gp = RandomFeatureGaussianProcess(
            in_features=hidden_size,
            out_features=out_features,
            learnable_lengthscale=False,
            learnable_outputscale=False,
            lengthscale=lengthscale,
            outputscale=outputscale,
            rank=rank)
    
    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        logits = self.gp(x)
        return logits

    def reinitialize_precision(self, device='cpu'):
        self.gp.reinitialize_precision(device=device)

    def update_precision_from_loader(self, train_loader, device='cpu'):
        self.eval()
        self.to(device)

        with torch.no_grad():
            for batch in train_loader:
                X = batch[0].to(device)
                h = self.fc_in(X)
                h = self.relu(h)
                h = self.res_blocks(h)
                self.gp.compute_batch_precision_update(h)


    def invert_covariance(self, device='cpu'):
        return self.gp.invert_covariance(device=device)

    def predict_proba(self, x, covariance, num_samples=100):
        x = self.fc_in(x)
        x = self.relu(x)
        x = self.res_blocks(x)
        proba = self.gp.predict_proba(x, covariance, num_samples=num_samples)
        return proba

