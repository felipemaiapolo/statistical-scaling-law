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
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from model import ch, sigmoid, extract_lower_triangle, reconstruct_corr_matrix

# Model fitting
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
    Monte Carlo log-likelihood for the heteroscedastic Beta mixed-effects model.
 
    For each observation i and outcome j, the model is
 
        Y_{ij} | alpha_{f(i)} ~ Beta(phi_{ij} * mu_{ij}, phi_{ij} * (1 - mu_{ij}))
 
    with
 
        mu_{ij}  = C_j + (1 - C_j) * sigmoid( (X_mean_i @ beta + alpha_{f(i)}) @ lambd + b )_j
        phi_{ij} = exp( X_phi_i^T @ phi_slope_{:, j} + phi_int_j )
 
    where the family-level random effect alpha_{f(i)} is K-dimensional Gaussian
    with Cholesky factor L (stored as its lower-triangular flat vector). The
    marginal likelihood is approximated by Monte Carlo: B standard-normal draws
    Z are transformed to alpha = Z @ L^T, the per-family log-likelihoods are
    averaged in the log-domain via logsumexp, and (optionally) summed across
    families.
 
    Parameters
    ----------
    X_mean : (n, p_mean) tensor
        Covariates entering the mean equation.
    X_phi : (n, p_phi) tensor
        Covariates entering the log-dispersion equation.
    Y : (n, J) tensor
        Outcomes in (0, 1). NaN entries are treated as missing and contribute 0
        to the log-likelihood.
    D : (n, F) tensor
        Family-membership indicator matrix mapping observations to families.
    C : (J,) or (1, J) tensor
        Lower bound (floor) for each outcome's mean, e.g. random-guessing rate.
    beta : (p_mean, K) tensor
        Mean-equation coefficients in latent factor space.
    lambd : (K, J) tensor
        Loading matrix from latent factors to the J outcomes.
    b : (1, J) tensor
        Per-outcome intercept inside the sigmoid.
    phi_int : (J,) tensor
        Per-outcome log-dispersion intercept.
    phi_slope : (p_phi, J) tensor
        Per-outcome log-dispersion slopes on X_phi.
    L : (K(K+1)/2,) tensor
        Flattened lower-triangular Cholesky factor of the random-effects
        covariance.
    K : int
        Latent factor dimension.
    Z : (B, K) tensor, optional
        Pre-drawn standard normal samples. If None, B fresh samples are drawn
        using `random_seed`.
    B : int, default 100000
        Number of Monte Carlo samples when Z is not supplied.
    eps : float, default 1e-6
        Clamp applied to mu to keep it inside (eps, 1 - eps).
    device : str, default 'cpu'
        Device for fresh Z draws when Z is None.
    random_seed : int, default 42
        Seed used when drawing Z internally.
    agg : bool, default True
        If True, return the summed log-likelihood (scalar). If False, return
        the per-family log-likelihood vector of length F.
 
    Returns
    -------
    torch.Tensor
        Scalar total log-likelihood if `agg=True`, otherwise per-family
        log-likelihoods.
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
    Fit the heteroscedastic Beta mixed-effects model by stochastic Monte Carlo
    maximum likelihood.
 
    The optimizer is Adam with a ReduceLROnPlateau schedule. The loadings
    matrix lambd is split into a non-negative diagonal block (lambd1, length K)
    and a free rectangular block (lambd2, shape K x (J-K)) to identify the
    model. The Cholesky factor L is re-normalized after each step so that the
    implied random-effects covariance has unit diagonal (a correlation matrix).
    Final parameter estimates are obtained by averaging the last
    `frac_hist_params` fraction of parameter snapshots, which smooths the
    stochastic-MLE iterates.
 
    Parameters
    ----------
    X_mean : (n, p_mean) tensor
        Mean-equation covariates.
    X_phi : (n, p_phi) tensor
        Log-dispersion covariates.
    Y : (n, J) tensor
        Outcomes in (0, 1); NaNs allowed.
    D : (n, F) tensor
        Family-indicator matrix.
    C : (J,) or (1, J) tensor
        Per-outcome floor for the mean.
    K : int, default 3
        Latent factor dimension.
    L, beta, b, phi_int, phi_slope, lambd1, lambd2 : tensors or None
        Optional starting values. If None each is drawn randomly; if provided
        each is perturbed by a small random draw (warm start with jitter).
    B : int, default 1000
        Number of Monte Carlo samples per evaluation.
    B_frequency : int, default 1
        Resample Z (and snapshot parameters) every `B_frequency` epochs.
    max_hist_params : int
        Keep at most this many parameter snapshots in memory.
    frac_hist_params : float, default 0.2
        Fraction of the *most recent* snapshots to average over when forming
        the final estimate.
    lr : float, default 1e-2
        Adam learning rate.
    scheduler_factor : float, default 0.9
        Multiplicative LR-decay factor on plateau.
    scheduler_patience : int, default 30
        Epochs with no improvement before LR is reduced.
    scheduler_threshold : float, default 1e-3
        Absolute improvement threshold for ReduceLROnPlateau.
    n_epochs : int, default 10000
        Maximum number of optimization steps.
    print_every : int, default 1000
        Logging interval.
    scale : float, default 1.0
        Standard deviation of random initialization draws.
    tol : float, default 1e-3
        Reserved for convergence checks (not currently used to early-stop).
    verbose : bool, default True
        Whether to show a progress bar and periodic logs.
    device : str
        Torch device.
    random_seed : int
        Seed for reproducible Z draws and initializations.
 
    Returns
    -------
    dict
        Dictionary with keys:
        - 'hist'       : list of per-epoch negative log-likelihood values.
        - 'loglike'    : final log-likelihood at the averaged parameters.
        - 'aic'        : Akaike Information Criterion at the averaged parameters.
        - 'grad_norm'  : per-parameter and total gradient norms at the optimum.
        - 'L', 'beta', 'b', 'phi_int', 'phi_slope', 'lambd1', 'lambd2', 'lambd' :
            averaged parameter tensors.
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
    
    """
    Convenience wrapper around `fit_model` implementing a heteroscedastic Beta
    mixed-effects scaling-law model with K latent factors.
 
    Attributes are populated by `fit`: after fitting, `beta`, `b`, `phi_int`,
    `phi_slope`, `lambd1`, `lambd2`, `lambd`, `sigma`, `L` hold the estimated
    parameters (as NumPy arrays), and `loglike`, `aic`, `gradnorm`, `outs`,
    `configs` summarize the fit and the random-restart sweep.
    """
    
    def __init__(self, K):

        """
        Initialize a scaling-law model.
 
        Parameters
        ----------
        K : int
            Number of latent factors.
        """
        
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
              reps = 10,
              B = 5000, #initial MC samples
              B_frequency = 1,
              lrs = [.05,.01,.005],
              scheduler_factors = [.99],  # Decay factor for line search learning rate
              n_epochs = 20000,
              scale = 1,
              tol = 1e-4,
              print_every = 1000,
              verbose = True,
              device='cpu'):

        """
        Run a random-restart grid over learning rates and scheduler factors,
        then keep the run with the highest log-likelihood.
 
        Each combination of (lr, scheduler_factor) is repeated `reps` times
        with a distinct seed, for a total of len(lrs) * len(scheduler_factors)
        * reps full fits. The best fit (by log-likelihood) populates the model
        attributes.
 
        Parameters
        ----------
        X_mean : array-like, shape (n, p_mean)
            Mean-equation covariates.
        Y : array-like, shape (n, J)
            Outcomes in (0, 1); NaNs are treated as missing.
        D : array-like, shape (n, F)
            Family-indicator matrix.
        C : array-like, shape (J,) or (1, J)
            Per-outcome floor.
        X_phi : array-like, shape (n, p_phi), optional
            Log-dispersion covariates. Defaults to X_mean if None.
        reps : int, default 10
            Number of random restarts per (lr, scheduler_factor) pair.
        B : int, default 5000
            Monte Carlo samples per evaluation.
        B_frequency : int, default 1
            Resampling frequency for Z (see `fit_model`).
        lrs : list of float
            Learning rates to try.
        scheduler_factors : list of float
            ReduceLROnPlateau decay factors to try.
        n_epochs : int, default 20000
            Maximum epochs per fit.
        scale : float, default 1
            Initialization scale.
        tol : float, default 1e-4
            Convergence tolerance forwarded to `fit_model`.
        print_every : int, default 1000
            Logging interval.
        verbose : bool, default True
            Toggle progress bars and periodic logs.
        device : str, default 'cpu'
            Torch device.
 
        Returns
        -------
        None
            Populates `self` in place. Notably:
                - self.outs       : list of all `fit_model` outputs.
                - self.configs    : dict of hyperparameters and log-likelihoods.
                - self.loglike    : best log-likelihood.
                - self.aic        : AIC of the best fit.
                - self.beta, self.b, self.phi_int, self.phi_slope,
                  self.lambd1, self.lambd2, self.lambd, self.L, self.sigma :
                  best-fit parameters as NumPy arrays.
        """
        
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


# Variance
def pack_theta(beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt):

    """
    Pack all free model parameters into a single 1-D tensor.
 
    The order is fixed: beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt.
    This is the inverse of `unpack_theta` and is used to feed
    `torch.autograd.functional.hessian` / `jacobian`, which expect a flat
    parameter vector.
 
    Parameters
    ----------
    beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt : tensors
        Model parameters with their natural shapes.
 
    Returns
    -------
    torch.Tensor
        Concatenated 1-D parameter vector.
    """
    
    return torch.cat([
        beta.contiguous().view(-1),
        lambd1.contiguous().view(-1),
        lambd2.contiguous().view(-1),
        b.contiguous().view(-1),
        phi_int.contiguous().view(-1),
        phi_slope.contiguous().view(-1),
        sigma_lt.contiguous().view(-1)
    ])

def unpack_theta(theta, shapes):

    """
    Split a flat parameter vector back into its component tensors.
 
    Inverse of `pack_theta`: given the flat vector `theta` and the original
    parameter shapes, reshape the slices back into
    (beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt).
 
    Parameters
    ----------
    theta : 1-D tensor
        Packed parameter vector produced by `pack_theta`.
    shapes : tuple of torch.Size
        Tuple of the original shapes, in the same order used by `pack_theta`:
        (beta_shape, lambd1_shape, lambd2_shape, b_shape, phi_int_shape,
        phi_slope_shape, sigma_lt_shape).
 
    Returns
    -------
    tuple of torch.Tensor
        (beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt) reshaped views
        into `theta`.
    """
    
    # shapes is a tuple: (beta_shape, lambd1_shape, lambd2_shape, b_shape, phi_int_shape, phi_slope_shape, sigma_lt_shape)
    beta_shape, lambd1_shape, lambd2_shape, b_shape, phi_int_shape, phi_slope_shape, sigma_lt_shape = shapes
    sizes = [torch.tensor(s).prod().item() for s in shapes]

    idx1 = sizes[0]
    idx2 = idx1 + sizes[1]
    idx3 = idx2 + sizes[2]
    idx4 = idx3 + sizes[3]
    idx5 = idx4 + sizes[4]
    idx6 = idx5 + sizes[5]

    beta_new      = theta[:idx1].view(beta_shape)
    lambd1_new    = theta[idx1:idx2].view(lambd1_shape)
    lambd2_new    = theta[idx2:idx3].view(lambd2_shape)
    b_new         = theta[idx3:idx4].view(b_shape)
    phi_int_new   = theta[idx4:idx5].view(phi_int_shape)
    phi_slope_new = theta[idx5:idx6].view(phi_slope_shape)
    sigma_lt_new  = theta[idx6:].view(sigma_lt_shape)

    return beta_new, lambd1_new, lambd2_new, b_new, phi_int_new, phi_slope_new, sigma_lt_new

def GetAsymptVar(X_mean, X_phi, Y, D, C, model, K,
                 device = 'cuda',
                 B = 25000,
                 lr = 1e-4,
                 grad_tol = 1e-3,
                 n_epochs = 100000,
                 compute_hessian = True,
                 reg = 1e-3,
                 random_seed=42):

    """
    Compute asymptotic-variance estimates for the MLE returned in `model`.
 
    Two estimators are produced (when applicable):
        - Inverse observed information: V_H = (H + reg * I)^{-1}, where H is
          the Hessian of the negative log-likelihood at theta_hat.
        - Inverse outer-product-of-gradients: V_J = (J^T J + reg * I)^{-1},
          where J is the Jacobian of the per-family log-likelihood
          contributions. This is the "OPG" / BHHH-style sandwich ingredient.
 
    Because the supplied parameters come from stochastic MLE iterates, the
    routine first runs a deterministic fine-tuning pass (Adam with a fixed
    Monte Carlo sample Z so the loss is a smooth function of theta) until the
    gradient norm falls below `grad_tol` or `n_epochs` is reached. The Hessian
    and Jacobian are then evaluated on CPU at the refined parameters.
 
    Standard errors are produced by extracting the diagonal of each
    variance matrix and taking square roots; they are returned grouped by
    parameter block.
 
    Parameters
    ----------
    X_mean, X_phi : array-like
        Mean and dispersion covariates.
    Y, D, C : array-like
        Outcomes, family indicators, and per-outcome floors.
    model : ScalingLaw
        Fitted model from which initial parameters are taken.
    K : int
        Latent dimension.
    device : str, default 'cuda'
        Device used for the fine-tuning phase. Hessian and Jacobian are
        always computed on CPU.
    B : int, default 25000
        Number of Monte Carlo samples held fixed during fine-tuning and
        variance evaluation.
    lr : float, default 1e-4
        Learning rate for the fine-tuning Adam optimizer.
    grad_tol : float, default 1e-3
        Early-stop threshold on the total gradient norm.
    n_epochs : int, default 100000
        Maximum fine-tuning epochs.
    compute_hessian : bool, default True
        If False, skip the Hessian-based estimator (which is the expensive one)
        and return only the Jacobian-based estimator.
    reg : float, default 1e-3
        Diagonal regularization added before inversion to ensure stability.
    random_seed : int, default 42
        Seed used to draw Z.
 
    Returns
    -------
    dict
        Dictionary with up to two keys:
        - 'H' : (when `compute_hessian=True`) a list `[info, ste_dict]` where
            `info` contains the Hessian H, its regularized inverse V, and the
            smallest eigenvalue of V; and `ste_dict` maps parameter names
            (suffixed with '_ste') to per-element standard errors.
        - 'J' : list `[info, ste_dict]` with the OPG-based V and matching SEs.
    """
    
    norm_dist = MultivariateNormal(torch.zeros(K), torch.eye(K))
    torch.manual_seed(random_seed)
    Z = norm_dist.sample(sample_shape=(B,)).to(device).float()
    beta = torch.tensor(model.beta,
                        device=device,
                        dtype=torch.float32,
                        requires_grad=True)
    lambd1 = torch.tensor(np.diag(model.lambd[:,:K]),
                          device=device,
                          dtype=torch.float32,
                          requires_grad=True)
    lambd2 = torch.tensor(model.lambd[:,K:],
                          device=device,
                          dtype=torch.float32,
                          requires_grad=True)
    b = torch.tensor(model.b,
                     device=device,
                     dtype=torch.float32,
                     requires_grad=True)
    phi_int = torch.tensor(model.phi_int,
                           device=device,
                           dtype=torch.float32,
                           requires_grad=True)
    phi_slope = torch.tensor(model.phi_slope,
                             device=device,
                             dtype=torch.float32,
                             requires_grad=True)
    sigma_lt = torch.tensor(extract_lower_triangle(torch.tensor(model.sigma)).clone().detach().numpy(),
                            device=device,
                            dtype=torch.float32,
                            requires_grad=True)
    param_shapes = (beta.shape, lambd1.shape, lambd2.shape, b.shape, phi_int.shape, phi_slope.shape, sigma_lt.shape)

    X_mean = torch.tensor(X_mean).to(device).float()
    X_phi  = torch.tensor(X_phi).to(device).float()
    Y = torch.tensor(Y).to(device).float()
    D = torch.tensor(D).to(device).float()
    C = torch.tensor(C).to(device).float()

    def loss_from_theta(theta, Z=Z, jac=False, device=device):

        """Evaluate (neg-)log-likelihood from the flat parameter vector theta.
 
        Reconstructs the Cholesky factor L of the correlation matrix from
        `sigma_lt`, rebuilds `lambd`, and calls `log_like`. If `jac=True`,
        returns the un-summed per-family log-likelihoods (signed positive)
        for Jacobian computation; otherwise returns the scalar negative
        log-likelihood.
        """
        
        beta_new, lambd1_new, lambd2_new, b_new, phi_int_new, phi_slope_new, sigma_lt_new = unpack_theta(theta, param_shapes)
        lt = torch.linalg.cholesky(reconstruct_corr_matrix(sigma_lt_new))
        mask = torch.tril(torch.ones_like(lt)).bool()
        L_new = -lt[mask]
        lambd_new = torch.hstack((torch.diag(lambd1_new), lambd2_new))
        loss = log_like(X_mean, X_phi, Y, D, C,
                         beta_new,
                         lambd_new,
                         b_new,
                         phi_int_new,
                         phi_slope_new,
                         L_new,
                         K,
                         Z,
                         agg=not jac)
        if jac:
            return loss
        else:
            return -loss

    ## Fine-tuning
    parameters = [sigma_lt, lambd1, lambd2, phi_int, phi_slope, b, beta]
    optimizer = Adam([
        {'params': sigma_lt,  'lr': lr},
        {'params': lambd1,    'lr': lr},
        {'params': lambd2,    'lr': lr},
        {'params': phi_int,   'lr': lr},
        {'params': phi_slope, 'lr': lr},
        {'params': b,         'lr': lr},
        {'params': beta,      'lr': lr}
    ])

    scheduler  = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=.999,
            patience=10,
            threshold=0,
            threshold_mode='abs'
        )
    grads = []
    losses = []
    for ep in tqdm(range(n_epochs), desc='fine-tuning'):
        optimizer.zero_grad()
        theta = pack_theta(beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt)
        loss = loss_from_theta(theta)
        loss.backward()
        losses.append(loss.item())

        # Compute gradient norm
        total_grad = torch.sqrt(sum((p.grad**2).sum() for p in parameters)).item()
        grads.append(total_grad)
        if ep%1000==0:
            print("grad norm:",grads[-1])
        if grads[-1]<grad_tol:
            break
        optimizer.step()
        scheduler.step(loss.item())

    plt.plot(losses)
    plt.ylabel('loss')
    plt.show()

    plt.plot(grads)
    plt.ylabel('grad')
    plt.yscale('log')
    plt.show()

    ## Computing variance
    device = 'cpu'
    beta = beta.cpu()
    lambd1 = lambd1.cpu()
    lambd2 = lambd2.cpu()
    b = b.cpu()
    phi_int = phi_int.cpu()
    phi_slope = phi_slope.cpu()
    sigma_lt = sigma_lt.cpu()
    Z = Z.cpu()
    X_mean = X_mean.cpu()
    X_phi  = X_phi.cpu()
    Y = Y.cpu()
    D = D.cpu()
    C = C.cpu()
    theta = pack_theta(beta, lambd1, lambd2, b, phi_int, phi_slope, sigma_lt)
    def loss_from_theta2(theta):
        return loss_from_theta(theta, Z=Z, device=device)
    def loss_from_theta3(theta):
        return loss_from_theta(theta, Z=Z, jac=True, device=device)

    out = {}

    if compute_hessian:
        H = hessian(loss_from_theta2, theta)
        H = (H+H.T)/2
        V = torch.linalg.inv(H + reg*np.eye(H.shape[0]))
        out["H"] = [{'min_eigenvalue':np.min(torch.linalg.eig(V).eigenvalues.cpu().numpy().astype(float)),'H':H.detach().cpu().numpy(),'V':V.detach().cpu().numpy()},{name+"_ste":np.sqrt(x.detach().cpu().numpy()) for x,name in zip(unpack_theta(torch.diag(V), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi_int', 'phi_slope', 'sigma_lt'])}]

    J = jacobian(loss_from_theta3, theta)
    V2 = torch.linalg.inv(J.T@J + reg*np.eye(J.T.shape[0]))
    out["J"] = [{'min_eigenvalue':np.min(torch.linalg.eig(V2).eigenvalues.cpu().numpy().astype(float)),'V':V2.detach().cpu().numpy()},{name+"_ste":np.sqrt(x.detach().cpu().numpy()) for x,name in zip(unpack_theta(torch.diag(V2), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi_int', 'phi_slope', 'sigma_lt'])}]

    ## Output
    return out

# Sampling
def SampleParams(beta, b, phi_int, phi_slope, lambd1, lambd2, sigma, cov, n=100):

    """
    Draw `n` parameter realizations from the asymptotic Normal approximation
    around the MLE.
 
    Parameters are packed via `pack_theta`, sampled jointly from
    N(theta_hat, cov), then unpacked. Each draw of sigma is projected onto the
    PSD cone (eigenvalue clipping at 0) and a small diagonal regularization is
    added; phi positivity is guaranteed by the exp() parameterization, so no
    projection is needed there.
 
    Parameters
    ----------
    beta, b, phi_int, phi_slope, lambd1, lambd2 : np.ndarray
        Point estimates of the corresponding parameters.
    sigma : np.ndarray, shape (K, K)
        Point estimate of the random-effects covariance / correlation matrix.
    cov : np.ndarray
        Asymptotic covariance matrix of the packed parameter vector (e.g.
        V_H or V_J from `GetAsymptVar`).
    n : int, default 100
        Number of draws.
 
    Returns
    -------
    tuple of lists
        (betas, bs, phi_ints, phi_slopes, lambds, sigmas) each of length n.
        `lambds[i]` is the reconstructed (K, J) loading matrix and `sigmas[i]`
        is the PSD-projected covariance matrix for draw i.
    """
    
    sigma_lt = extract_lower_triangle(sigma)
    mu = pack_theta(torch.tensor(beta), torch.tensor(lambd1), torch.tensor(lambd2), torch.tensor(b), torch.tensor(phi_int), torch.tensor(phi_slope), torch.tensor(sigma_lt)).numpy()
    param_shapes = (beta.shape, lambd1.shape, lambd2.shape, b.shape, phi_int.shape, phi_slope.shape, sigma_lt.shape)
    thetas = np.random.multivariate_normal(mu, cov, size=n)
    thetas = [{name:x.detach().cpu().numpy() for x, name in zip(unpack_theta(torch.tensor(t), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi_int', 'phi_slope', 'sigma_lt'])} for t in thetas]

    betas=[]
    bs=[]
    phi_ints=[]
    phi_slopes=[]
    lambds=[]
    sigmas=[]
    for i in range(len(thetas)):
        beta      = thetas[i]['beta']
        b         = thetas[i]['b']
        phi_int   = thetas[i]['phi_int']
        phi_slope = thetas[i]['phi_slope']
        lambd = np.hstack((np.diag(thetas[i]['lambd1']), thetas[i]['lambd2']))
        sigma = reconstruct_corr_matrix(torch.tensor(thetas[i]['sigma_lt'])).numpy()

        V,U=np.linalg.eigh(sigma)
        sigma = U@np.diag([v if v > 0 else 0 for v in V])@U.T #projection on PSD
        eps = 1e-4
        sigma += eps*np.eye(sigma.shape[0]) #regularization / non-degenerate distribution
        # phi positivity is guaranteed by exp() - no projection needed

        betas.append(beta)
        bs.append(b)
        phi_ints.append(phi_int)
        phi_slopes.append(phi_slope)
        lambds.append(lambd)
        sigmas.append(sigma)
        
    return betas, bs, phi_ints, phi_slopes, lambds, sigmas


def log_like_fe(X_mean, X_phi, Y, D, C, A, beta, lambd, b, phi_int, phi_slope):

    """
    Conditional ("fixed-effects" form) log-likelihood given the latent A.
 
    Same Beta likelihood as `log_like`, but with the family-level random
    effects A treated as known rather than integrated out. Used inside
    `joint_dist_one_fam` for posterior inference / MCMC over A. NaN entries
    in Y contribute zero.
 
    Parameters
    ----------
    X_mean : (n, p_mean) tensor
    X_phi : (n, p_phi) tensor
    Y : (n, J) tensor
    D : (n, F) tensor
        Family-indicator matrix, used to attribute each row's random effect.
    C : (J,) tensor
        Per-outcome floor.
    A : (F, K) tensor
        Family-level random effects (treated as fixed).
    beta : (p_mean, K) tensor
    lambd : (K, J) tensor
    b : (1, J) tensor
    phi_int : (J,) tensor
    phi_slope : (p_phi, J) tensor
 
    Returns
    -------
    torch.Tensor
        Scalar conditional log-likelihood, summed over observations and outcomes.
 
    Notes
    -----
    Uses an `eps` variable that must be defined in the calling scope (the
    function clamps mu via `mu.clamp(min=eps, max=1.0 - eps)`).
    """
    
    mu = C+(1-C)*sigmoid((X_mean@beta+D@A)@lambd+b)
    mu = mu.clamp(min=eps, max=1.0 - eps) 
    # Heteroscedastic dispersion: phi_ij = exp(x_phi_i^T phi_slope_j + phi_int_j)
    log_phi = X_phi @ phi_slope + phi_int         # (n, J)
    phi_ij  = torch.exp(log_phi).clamp(max=1e6)   # (n, J), cap for numerical safety
    beta_dist = Beta(phi_ij*mu, phi_ij*(1-mu))
    nan_mask = torch.isnan(Y) #dealing with missing values
    Y_clipped = torch.where(nan_mask, torch.full_like(Y, 0.5), Y)
    ll_terms = beta_dist.log_prob(Y_clipped[None, :])
    loglike = torch.where(nan_mask, torch.zeros_like(ll_terms), ll_terms).sum()
    return loglike

def joint_dist_one_fam(X_mean, X_phi, Y, C, A, beta, lambd, b, phi_int, phi_slope, sigma, device='cpu'):

    """
    Unnormalized log-posterior of A for a single family.
 
    Combines the conditional log-likelihood (`log_like_fe`) with the Gaussian
    prior log p(A | sigma) = MVN(0, sigma) and returns their sum, i.e. the
    target log-density for posterior inference on a single family's random
    effect. The data passed in should already be filtered to that family
    (so D is implicitly the identity here).
 
    Parameters
    ----------
    X_mean, X_phi, Y : tensors
        Family-restricted covariates and outcomes.
    C : (J,) tensor
        Per-outcome floor.
    A : (K,) tensor
        Random-effect vector for this family.
    beta, lambd, b, phi_int, phi_slope : tensors
        Population-level parameters.
    sigma : (K, K) tensor
        Prior covariance of A.
    device : str, default 'cpu'
        Device on which to evaluate.
 
    Returns
    -------
    torch.Tensor
        Scalar log p(Y_fam, A | params, sigma).
    """
    
    D = torch.eye(1).to(device)
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(A.squeeze().shape[0], device=device),
        covariance_matrix=sigma.to(device)
    )
    return log_like_fe(X_mean.to(device), X_phi.to(device), Y.to(device), D.to(device), C.to(device), A.to(device).squeeze()[None,:], beta.to(device), lambd.to(device), b.to(device), phi_int.to(device), phi_slope.to(device)) + prior.log_prob(A.to(device).squeeze()).to(device)

def metropolis_hastings(init,
                        sigma_prop,
                        X_mean_tensor,
                        X_phi_tensor,
                        Y_tensor,
                        C_tensor,
                        beta_tensor,
                        lambd_tensor,
                        b_tensor,
                        phi_int_tensor,
                        phi_slope_tensor,
                        sigma_tensor,
                        n_samples=2000,
                        burn_in=100,
                        thinning=10):
    
    """
    Random-walk Metropolis-Hastings sampler for the random-effect A of a
    single family.
 
    The target log-density is `joint_dist_one_fam` (the unnormalized
    log-posterior of A). Proposals are Gaussian with covariance `sigma_prop`,
    drawn efficiently via a Cholesky decomposition. Acceptance is computed in
    log-space for numerical stability.
 
    The sampler performs `thinning` proposal steps between recorded states,
    discards the first `burn_in` recorded states, and returns the next
    `n_samples` recorded states.
 
    Parameters
    ----------
    init : (K,) tensor
        Initial state of the chain (often the MAP of A).
    sigma_prop : (K, K) tensor
        Covariance matrix of the Gaussian proposal (often a Laplace
        approximation to the posterior).
    X_mean_tensor, X_phi_tensor, Y_tensor, C_tensor : tensors
        Family-restricted data forwarded to `joint_dist_one_fam`.
    beta_tensor, lambd_tensor, b_tensor, phi_int_tensor, phi_slope_tensor,
        sigma_tensor : tensors
        Population-level parameters forwarded to `joint_dist_one_fam`.
    n_samples : int, default 2000
        Number of post-burn-in samples to return.
    burn_in : int, default 100
        Number of initial recorded states to discard.
    thinning : int, default 10
        Number of proposal steps between recorded states.
 
    Returns
    -------
    torch.Tensor of shape (n_samples, K)
        Posterior draws of A.
    """
    
    def log_density(A):
        """Target log-density: log p(Y_fam, A | params, sigma)."""
        return joint_dist_one_fam(X_mean_tensor, X_phi_tensor, Y_tensor, C_tensor,
                                  A, beta_tensor, lambd_tensor,
                                  b_tensor, phi_int_tensor, phi_slope_tensor, sigma_tensor)

    d = len(init)
    samples = torch.zeros((n_samples, d), dtype=init.dtype, device=init.device)
    current = init.clone()
    current_log_density = log_density(current)

    cov_chol = torch.linalg.cholesky(sigma_prop)  # Cholesky decomposition for efficient sampling

    collected = 0
    i = 0
    while collected < n_samples:
        for _ in range(thinning):
            proposal = current + cov_chol @ torch.randn(d, dtype=init.dtype, device=init.device)
            proposal_log_density = log_density(proposal)

            # Compute log acceptance probability
            log_alpha = proposal_log_density - current_log_density

            if torch.log(torch.rand((), dtype=init.dtype, device=init.device)) < log_alpha:
                current = proposal
                current_log_density = proposal_log_density

        if i >= burn_in:
            samples[collected] = current
            collected += 1
        i += 1

    return samples


def SampleAlpha(i,
                 betas,
                 lambds,
                 bs,
                 phi_ints,
                 phi_slopes,
                 sigmas,
                 D,
                 fam_number,
                 X_mean,
                 X_phi,
                 Y,
                 C,
                 K,
                 random_seed=42,
                 n_samples=10,
                 burn_in=100,
                 thinning=30,
                 return_map=False):

    """
    Sample the latent random effect A for one family, using a Laplace-aided
    Metropolis-Hastings scheme.
 
    The procedure is:
        1. Restrict the data to the family identified by `fam_number`.
        2. Find the MAP of A by minimizing the negative joint log-density
           with LBFGS (strong-Wolfe line search), starting from zero.
        3. Approximate the posterior curvature by the inverse Hessian of the
           negative log-density at the MAP (Laplace covariance).
        4. Either return the MAP directly (`return_map=True`) or run
           `metropolis_hastings` with proposal covariance equal to the
           Laplace covariance.
 
    Parameter samples (`betas`, `lambds`, `bs`, `phi_ints`, `phi_slopes`,
    `sigmas`) are typically drawn from `SampleParams`, so that posterior
    uncertainty in A can be propagated together with parameter uncertainty.
 
    Parameters
    ----------
    i : int
        Index into the parameter sample lists; selects which posterior draw of
        the population parameters to condition on.
    betas, lambds, bs, phi_ints, phi_slopes, sigmas : lists of array-likes
        Posterior draws of the population parameters (one entry per draw).
    D : array-like, shape (n, F)
        Family-indicator matrix for the full dataset.
    fam_number : int
        Column index of D selecting the family to sample.
    X_mean, X_phi, Y : array-likes
        Full-dataset covariates and outcomes.
    C : array-like
        Per-outcome floor.
    K : int
        Latent dimension of A.
    random_seed : int, default 42
        Seed for reproducibility.
    n_samples : int, default 10
        Number of post-burn-in MH samples to return.
    burn_in : int, default 100
        MH burn-in.
    thinning : int, default 30
        MH thinning interval.
    return_map : bool, default False
        If True, skip MH and return only the MAP estimate (as a torch.Tensor).
 
    Returns
    -------
    np.ndarray of shape (n_samples, K)
        Posterior draws of A for the selected family (when `return_map=False`).
    torch.Tensor of shape (K,)
        MAP estimate of A (when `return_map=True`).
    """

    torch.manual_seed(random_seed)

    # Select family
    ind_fam = D[:, fam_number] == 1
    X_mean_tensor    = torch.tensor(X_mean, requires_grad=False)[ind_fam].float()
    X_phi_tensor     = torch.tensor(X_phi,  requires_grad=False)[ind_fam].float()
    Y_tensor         = torch.tensor(Y, requires_grad=False)[ind_fam].float()
    C_tensor         = torch.tensor(C, requires_grad=False).float()
    beta_tensor      = torch.tensor(betas[i], requires_grad=False).float()
    lambd_tensor     = torch.tensor(lambds[i], requires_grad=False).float()
    b_tensor         = torch.tensor(bs[i], requires_grad=False).float()
    phi_int_tensor   = torch.tensor(phi_ints[i], requires_grad=False).float()
    phi_slope_tensor = torch.tensor(phi_slopes[i], requires_grad=False).float()
    sigma_tensor     = torch.tensor(sigmas[i], requires_grad=False).float()

    # Optimize A_map
    A_map = torch.zeros(K, requires_grad=True).float()
    def loss_function(A_var):
        return -joint_dist_one_fam(X_mean_tensor, X_phi_tensor, Y_tensor, C_tensor,
                                   A_var, beta_tensor, lambd_tensor,
                                   b_tensor, phi_int_tensor, phi_slope_tensor, sigma_tensor)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(A_map)
        loss.backward()
        return loss
    optimizer = torch.optim.LBFGS([A_map], lr=1, max_iter=1000,
                                  history_size=100, line_search_fn="strong_wolfe")
    optimizer.step(closure)

    A_sigma = torch.linalg.inv(
        torch.autograd.functional.hessian(loss_function, A_map)
    ).float()

    A_map, A_sigma = A_map.float().detach().cpu(), A_sigma.detach().cpu()

    if return_map:
        return A_map

    # Sample alphas
    return metropolis_hastings(
        A_map, A_sigma, X_mean_tensor, X_phi_tensor, Y_tensor, C_tensor,
        beta_tensor, lambd_tensor, b_tensor, phi_int_tensor, phi_slope_tensor, sigma_tensor,
        n_samples=n_samples, burn_in=burn_in, thinning=thinning
    ).cpu().numpy()