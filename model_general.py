# Standard library imports
import numpy as np

# Third-party imports
import torch
from torch import nn
from torch.nn import Parameter
from torch.distributions import Beta, MultivariateNormal, Normal
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.autograd.functional import hessian, jacobian
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from joblib import Parallel, delayed

# Aliases / constants
sigmoid = torch.sigmoid

def ch(x: torch.Tensor, tril: bool = True) -> torch.Tensor:
    """
    Build a lower-triangular matrix (or its product) from a flattened tensor.
    """
    n = int((2 * len(x)) ** 0.5)
    if (n * (n + 1)) // 2 != len(x):
        raise ValueError("Length of x is not suitable to fill a lower triangular matrix")

    result = torch.zeros((n, n), dtype=x.dtype, device=x.device)
    row_idx, col_idx = torch.tril_indices(n, n)
    result[row_idx, col_idx] = x

    return result if tril else result @ result.T

def log_like(
    X_mean: torch.Tensor,
    X_phi: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    beta: torch.Tensor,
    lambd: torch.Tensor,
    b: torch.Tensor,
    phi_int: torch.Tensor,
    phi_slope: torch.Tensor,
    L: torch.Tensor,
    K: int,
    Z: torch.Tensor = None,
    B: int = 100000,
    eps: float = 1e-6,
    device: str = 'cpu',
    random_seed = 42,
    agg=True
) -> torch.Tensor:
    """
    Compute the Monte Carlo log-likelihood under the specified model.
    The mean uses X_mean with covariate matrix beta of shape (p_mean, K).
    Dispersion is parameterized as phi_ij = exp(x_phi_i^T phi_slope_j + phi_int_j),
    where phi_slope has shape (p_phi, J).
    """
    if Z is not None:
        B = Z.shape[0]
    else:
        torch.manual_seed(random_seed)
        norm_dist = MultivariateNormal(torch.zeros(K), torch.eye(K))
        Z = norm_dist.sample(sample_shape=(B,)).to(device)

    A = Z @ ch(L).T
    logits = (X_mean @ beta)[None, :, :] + A[:, None, :]
    logits = logits @ lambd + b[None, :]

    mu = C[None, :] + (1 - C[None, :]) * sigmoid(logits)
    mu = mu.clamp(min=eps, max=1.0 - eps)

    # Heteroscedastic dispersion: phi_ij = exp(x_phi_i^T phi_slope_j + phi_int_j)
    # phi_int:   shape (J,)
    # phi_slope: shape (p_phi, J)
    # X_phi:     shape (n, p_phi)
    log_phi = X_phi @ phi_slope + phi_int         # (n, J)
    phi_ij  = torch.exp(log_phi).clamp(max=1e6)   # (n, J), cap for numerical safety
    beta_dist = Beta(phi_ij[None, :, :] * mu,
                     phi_ij[None, :, :] * (1 - mu))

    nan_mask = torch.isnan(Y) #dealing with missing values
    Y_clipped = torch.where(nan_mask, torch.full_like(Y, 0.5), Y)
    ll_terms = beta_dist.log_prob(Y_clipped[None, :])
    ll_terms = torch.where(nan_mask, torch.zeros_like(ll_terms), ll_terms).sum(-1)

    loglike = (
        torch.logsumexp(ll_terms @ D, dim=0)
        - np.log(B)
    )
    if agg:
        return loglike.sum()
    else:
        return loglike

def fit_model(
    X_mean: torch.Tensor,
    X_phi: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    K: int = 3,
    L: torch.Tensor = None,
    beta: torch.Tensor = None,
    b: torch.Tensor = None,
    phi_int: torch.Tensor = None,
    phi_slope: torch.Tensor = None,
    lambd1: torch.Tensor = None,
    lambd2: torch.Tensor = None,
    B: int = 1000,              # initial MC samples
    B_frequency: int = 1,
    max_hist_params = 99999,
    frac_hist_params = .2,
    lr: float = 1e-2,
    scheduler_factor: float = .9,    # reduction factor on plateau
    scheduler_patience: int = 30,     # epochs with no improvement before reducing
    scheduler_threshold: float = 1e-3,
    n_epochs: int = 10000,
    print_every: int = 1000,
    scale: float = 1.,
    tol: float = 1e-3,
    verbose: bool = True,
    device: str = 'cpu',
    random_seed: int = 42,
) -> dict:
    """
    Fit the mixed-effects model via stochastic optimization.
    Returns history, AIC, best parameters, and convergence flag.
    The mean uses X_mean (shape (n, p_mean)) and the dispersion uses X_phi
    (shape (n, p_phi)). Dispersion is phi_ij = exp(x_phi_i^T phi_slope_j + phi_int_j).
    """
    torch.manual_seed(random_seed)

    N      = D.shape[1]
    p_mean = X_mean.shape[1]
    p_phi  = X_phi.shape[1]
    J      = Y.shape[1]

    # Helper for random initialization
    def _rand(shape, func, abs_val=False):
        t = func(0, scale, size=shape, device=device)
        return torch.abs(t) if abs_val else t

    # Initialize or perturb parameters
    L_rand         = _rand(((K * (K + 1)) // 2,), torch.normal)
    beta_rand      = _rand((p_mean, K), torch.normal)
    b_rand         = _rand((1, J), torch.normal)
    phi_int_rand   = _rand((J,),    torch.normal)             # unconstrained (log-scale)
    phi_slope_rand = _rand((p_phi, J), torch.normal) * 0.01   # near-zero => warm start near homoscedastic
    l1_rand        = _rand((K,), torch.normal, abs_val=True)
    l2_rand        = _rand((K, J - K), torch.normal)

    def _init(param, rand):
        if param is None:
            return Parameter(rand)
        data = param.cpu().detach().numpy() + rand.cpu().detach().numpy()
        return Parameter(torch.tensor(data, device=device))

    L         = _init(L,         L_rand)
    beta      = _init(beta,      beta_rand)
    b         = _init(b,         b_rand)
    phi_int   = _init(phi_int,   phi_int_rand)
    phi_slope = _init(phi_slope, phi_slope_rand)
    lambd1    = _init(lambd1,    l1_rand)
    lambd2    = _init(lambd2,    l2_rand)

    # Optimizer setup
    parameters = [L, lambd1, lambd2, phi_int, phi_slope, b, beta]
    optimizer  = Adam(parameters, lr=lr)
    scheduler  = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        threshold_mode='abs'
    )

    hist = []
    hist_params = []

    for ep in tqdm(range(n_epochs), desc="Training ME model", disable=not verbose):
        # Resample Z periodically
        if ep % B_frequency == 0:
            norm_dist = MultivariateNormal(torch.zeros(K), torch.eye(K))
            Z = norm_dist.sample(sample_shape=(B,)).to(device)

            # Keeping track
            current_params = {
                'L':         L.detach().cpu().clone(),
                'lambd1':    lambd1.detach().cpu().clone(),
                'lambd2':    lambd2.detach().cpu().clone(),
                'phi_int':   phi_int.detach().cpu().clone(),
                'phi_slope': phi_slope.detach().cpu().clone(),
                'b':         b.detach().cpu().clone(),
                'beta':      beta.detach().cpu().clone(),
            }
            hist_params.append(current_params)
            if len(hist_params)>max_hist_params:
                hist_params = hist_params[-max_hist_params:]

        # Build full lambda matrix
        lambd = torch.hstack((torch.diag(lambd1), lambd2))

        optimizer.zero_grad()
        loss = -log_like(
            X_mean=X_mean, X_phi=X_phi, Y=Y, D=D, C=C,
            beta=beta, lambd=lambd,
            b=b, phi_int=phi_int, phi_slope=phi_slope,
            L=L, K=K,
            Z=Z, device=device
        )
        loss /= (Y.shape[0] * Y.shape[1])
        hist.append(loss.item())

        # Backprop
        loss.backward()

        # Compute gradient norm
        total_grad = torch.sqrt(sum((p.grad**2).sum() for p in parameters)).item()
        grad_info = {name: torch.norm(param.grad).item()
                     for name, param in zip(
                         ['L','lambd1','lambd2','phi_int','phi_slope','b','beta'], parameters)}

        # Logging
        if verbose and ((ep+1) % print_every == 0):
            lr_now = scheduler.get_last_lr()[0]
            print(f"epoch={ep+1:5d}, B={B:5d}, grad={total_grad:.5f}, "
                  f"loss={loss.item():.5f}, lr={lr_now:.5f}")
            print(" grads:", grad_info)

        optimizer.step()
        scheduler.step(loss.item())

        # Projection step
        with torch.no_grad():
            # Keep L lower-triangular with unit row norms
            L.data = ch(L.data, tril=True)
            row_norms = L.data.norm(dim=1, keepdim=True)
            L.data = L.data / row_norms              # still full matrix
            L.data = L.data[torch.tril(torch.ones_like(L)).bool()]

            # phi positivity is guaranteed by exp() - no clamp needed
            lambd1.clamp_(min=0)

    # Finalize best parameters
    n_avg = int(frac_hist_params*len(hist_params))
    avg_params = {}
    for key in hist_params[-1]:
        avg_params[key] = [d[key].numpy() for d in hist_params[-n_avg:]]
        avg_params[key] = torch.tensor(np.mean(np.array(avg_params[key]),0))

    avg_params['lambd'] = torch.hstack((torch.diag(avg_params['lambd1']), avg_params['lambd2']))

    # Compute AIC, loglike, and grad
    device = 'cpu'

    n_params = int(K*(K-1)/2 +
                   sum(int(np.prod(avg_params[p].shape))
                       for p in ['beta','lambd1','lambd2','b','phi_int','phi_slope']))

    beta      = avg_params['beta'].clone().detach().requires_grad_(True)
    lambd1    = avg_params['lambd1'].clone().detach().requires_grad_(True)
    lambd2    = avg_params['lambd2'].clone().detach().requires_grad_(True)
    b         = avg_params['b'].clone().detach().requires_grad_(True)
    phi_int   = avg_params['phi_int'].clone().detach().requires_grad_(True)
    phi_slope = avg_params['phi_slope'].clone().detach().requires_grad_(True)
    L         = avg_params['L'].clone().detach().requires_grad_(True)
    lambd     = torch.hstack((torch.diag(lambd1), lambd2))

    ll = log_like(
        X_mean=X_mean.to(device), X_phi=X_phi.to(device),
        Y=Y.to(device), D=D.to(device), C=C.to(device),
        beta=beta.to(device),
        lambd=lambd.to(device),
        b=b.to(device),
        phi_int=phi_int.to(device),
        phi_slope=phi_slope.to(device),
        L=L.to(device),
        K=K, device=device)

    ll.backward()

    cte = int(Y.shape[0]*Y.shape[1])
    grad_norm = {}
    grad_norm['beta']      = beta.grad.norm()/cte
    grad_norm['lambd1']    = lambd1.grad.norm()/cte
    grad_norm['lambd2']    = lambd2.grad.norm()/cte
    grad_norm['b']         = b.grad.norm()/cte
    grad_norm['phi_int']   = phi_int.grad.norm()/cte
    grad_norm['phi_slope'] = phi_slope.grad.norm()/cte
    grad_norm['L']         = L.grad.norm()/cte
    grad_norm['total']     = (beta.grad.norm()**2
                              + lambd1.grad.norm()**2
                              + lambd2.grad.norm()**2
                              + b.grad.norm()**2
                              + phi_int.grad.norm()**2
                              + phi_slope.grad.norm()**2
                              + L.grad.norm()**2)**.5 / cte

    print("final grad norm:", grad_norm['total'].item())

    aic = -2 * (ll.item() - n_params)

    return {
        'hist': hist,
        'loglike': ll.item(),
        'aic': aic,
        #'hist_params': hist_params,
        'grad_norm': grad_norm,
        **avg_params
    }

class ScalingLaw:
    def __init__(self, K):
        self.K = K
        self.L = None
        self.beta = None
        self.b = None
        self.phi_int = None
        self.phi_slope = None
        self.lambd1 = None
        self.lambd2 = None

    def fit(self,
              X_mean,
              Y,
              D,
              C,
              X_phi = None,                 # if None, defaults to X_mean
              fe_start = False,
              reps = 10,
              B = 5000, #initial MC samples
              B_frequency = 1,
              lrs = [.05,.01,.005],
              scheduler_factors = [.99],  # Decay factor for line search learning rate
              n_epochs = 20000,
              lr_fe = .1,
              scheduler_factor_fe = .9999,
              n_epochs_fe = 15000,
              scale = 1,
              tol = 1e-4,
              print_every = 1000,
              verbose = True,
              device='cpu'):

        X_mean = torch.tensor(X_mean).float().to(device)
        if X_phi is None:
            X_phi = X_mean
        else:
            X_phi = torch.tensor(X_phi).float().to(device)
        Y = torch.tensor(Y).float().to(device)
        D = torch.tensor(D).float().to(device)
        C = torch.tensor(C).float().to(device)

        outs = []
        configs = {'random_seed':[], 'lr':[], 'scheduler_factor':[], 'loglike':[]}
        r=0
        for lr in tqdm(lrs, desc="Different lrs"):
            for scheduler_factor in tqdm(scheduler_factors, desc="Different scheduler factors"):
                for _ in tqdm(range(reps), desc="Reps"):
                    outs.append(fit_model(X_mean,
                                          X_phi,
                                          Y,
                                          D,
                                          C,
                                          K = self.K,
                                          L = self.L,
                                          beta = self.beta,
                                          b = self.b,
                                          phi_int = self.phi_int,
                                          phi_slope = self.phi_slope,
                                          lambd1 = self.lambd1,
                                          lambd2 = self.lambd2,
                                          B = B, #initial MC samples
                                          lr = lr,
                                          scheduler_factor = scheduler_factor,
                                          n_epochs = n_epochs,
                                          scale = scale,
                                          tol = tol,
                                          verbose = verbose,
                                          print_every = print_every,
                                          device = device,
                                          random_seed = r))
                    configs['loglike'].append(outs[-1]['loglike'])
                    configs['random_seed'].append(r)
                    configs['lr'].append(lr)
                    configs['scheduler_factor'].append(scheduler_factor)
                    r+=1

        self.configs = configs
        self.outs = outs
        ind = np.argmax([o['loglike'] for o in self.outs])
        self.loglike   = self.outs[ind]['loglike']
        self.gradnorm  = self.outs[ind]['grad_norm']
        self.aic       = self.outs[ind]['aic']
        self.sigma     = ch(self.outs[ind]['L'], tril=False).cpu().numpy()
        self.L         = self.outs[ind]['L'].cpu().numpy()
        self.lambd     = self.outs[ind]['lambd'].cpu().numpy()
        self.phi_int   = self.outs[ind]['phi_int'].cpu().numpy()
        self.phi_slope = self.outs[ind]['phi_slope'].cpu().numpy()
        self.b         = self.outs[ind]['b'].cpu().numpy()
        self.beta      = self.outs[ind]['beta'].cpu().numpy()