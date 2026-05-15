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
    X: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    beta: torch.Tensor,
    lambd: torch.Tensor,
    b: torch.Tensor,
    phi: torch.Tensor,
    L: torch.Tensor,
    K: int,
    Z: torch.Tensor = None,
    B: int = 100000,
    device: str = 'cpu',
    random_seed = 42,
    agg=True
) -> torch.Tensor:
    """
    Compute the Monte Carlo log-likelihood under the specified model.
    """
    if Z is not None:
        B = Z.shape[0]
    else:
        torch.manual_seed(random_seed)
        norm_dist = MultivariateNormal(torch.zeros(K), torch.eye(K))
        Z = norm_dist.sample(sample_shape=(B,)).to(device)

    A = Z @ ch(L).T
    logits = (X @ beta)[None, :, :] + A[:, None, :]
    logits = torch.clamp(logits @ lambd + b[None, :], min=-4.5, max=4.5)

    mu = C[None, :] + (1 - C[None, :]) * sigmoid(logits)
    beta_dist = Beta(phi[None, :] * mu, phi[None, :] * (1 - mu))

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
    X: torch.Tensor,
    Y: torch.Tensor,
    D: torch.Tensor,
    C: torch.Tensor,
    K: int = 3,
    L: torch.Tensor = None,
    beta: torch.Tensor = None,
    b: torch.Tensor = None,
    phi: torch.Tensor = None,
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
    """
    torch.manual_seed(random_seed)

    N = D.shape[1]
    p = X.shape[1]
    J = Y.shape[1]

    # Helper for random initialization
    def _rand(shape, func, abs_val=False):
        t = func(0, scale, size=shape, device=device)
        return torch.abs(t) if abs_val else t

    # Initialize or perturb parameters
    L_rand    = _rand(((K * (K + 1)) // 2,), torch.normal)
    beta_rand = _rand((p, K), torch.normal)
    b_rand    = _rand((1, J), torch.normal)
    phi_rand  = _rand((1, J), torch.normal, abs_val=True)
    l1_rand   = _rand((K,), torch.normal, abs_val=True)
    l2_rand   = _rand((K, J - K), torch.normal)

    def _init(param, rand):
        if param is None:
            return Parameter(rand)
        data = param.cpu().detach().numpy() + rand.cpu().detach().numpy()
        return Parameter(torch.tensor(data, device=device))

    L      = _init(L,    L_rand)
    beta   = _init(beta, beta_rand)
    b      = _init(b,    b_rand)
    phi    = _init(phi,  phi_rand)
    lambd1 = _init(lambd1, l1_rand)
    lambd2 = _init(lambd2, l2_rand)

    # Optimizer setup
    parameters = [L, lambd1, lambd2, phi, b, beta]
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
                'L':     L.detach().cpu().clone(),
                'lambd1':lambd1.detach().cpu().clone(),
                'lambd2':lambd2.detach().cpu().clone(),
                'phi':   phi.detach().cpu().clone(),
                'b':     b.detach().cpu().clone(),
                'beta':  beta.detach().cpu().clone(),
            }
            hist_params.append(current_params)
            if len(hist_params)>max_hist_params:
                hist_params = hist_params[-max_hist_params:]
            
        # Build full lambda matrix
        lambd = torch.hstack((torch.diag(lambd1), lambd2))

        optimizer.zero_grad()
        loss = -log_like(
            X=X, Y=Y, D=D, C=C,
            beta=beta, lambd=lambd,
            b=b, phi=phi, L=L, K=K,
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
                         ['L','lambd1','lambd2','phi','b','beta'], parameters)}

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

            # Enforce non-negativity
            phi.clamp_(min=0)
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
                       for p in ['beta','lambd1','lambd2','b','phi']))
    
    beta   = avg_params['beta'].clone().detach().requires_grad_(True)
    lambd1 = avg_params['lambd1'].clone().detach().requires_grad_(True)
    lambd2 = avg_params['lambd2'].clone().detach().requires_grad_(True)
    b      = avg_params['b'].clone().detach().requires_grad_(True)
    phi    = avg_params['phi'].clone().detach().requires_grad_(True)
    L      = avg_params['L'].clone().detach().requires_grad_(True)
    lambd = torch.hstack((torch.diag(lambd1), lambd2))
    
    ll = log_like(
        X=X.to(device), Y=Y.to(device),
        D=D.to(device), C=C.to(device),
        beta=beta.to(device),
        lambd=lambd.to(device),
        b=b.to(device),
        phi=phi.to(device),
        L=L.to(device),
        K=K, device=device)

    ll.backward()

    cte = int(Y.shape[0]*Y.shape[1])
    grad_norm = {}
    grad_norm['beta']=beta.grad.norm()/cte
    grad_norm['lambd1']=lambd1.grad.norm()/cte
    grad_norm['lambd2']=lambd2.grad.norm()/cte
    grad_norm['b']=b.grad.norm()/cte
    grad_norm['phi']=phi.grad.norm()/cte
    grad_norm['L']=L.grad.norm()/cte
    grad_norm['total']=(beta.grad.norm()**2+lambd1.grad.norm()**2+lambd2.grad.norm()**2+b.grad.norm()**2+phi.grad.norm()**2+L.grad.norm()**2)**.5/cte

    print("final grad norm:",grad_norm['total'].item())
    
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
        self.phi = None
        self.lambd1 = None
        self.lambd2 = None
   
    def fit(self,
              X,
              Y,
              D,
              C,
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
        
        X = torch.tensor(X).float().to(device)
        Y = torch.tensor(Y).float().to(device)
        D = torch.tensor(D).float().to(device)
        C = torch.tensor(C).float().to(device)

        if fe_start:

            #model_starts = []
            #for lr in tqdm(lrs, desc="Different lrs"):
            #    for scheduler_factor in tqdm(scheduler_factors, desc="Different scheduler factors"):
            #model_starts.append(ScalingLawFixedEffects(self.K))
            #model_starts[-1].fit(X, Y, D, C, lr=lr, n_epochs=n_epochs_fe, scheduler_factor = scheduler_factor, verbose=verbose, device=device)
            #losses = [m.hist[-1] for m in model_starts]
            #self.model_start = model_starts[np.argmin(losses)]

            self.model_start = ScalingLawFixedEffects(self.K)
            self.model_start.fit(X, Y, D, C, lr=lr_fe, n_epochs=n_epochs_fe, scheduler_factor = scheduler_factor_fe, verbose=verbose, device=device)
            
            corr = torch.corrcoef(self.model_start.A.cpu().detach().T)
            if self.K>1: 
                L = torch.linalg.cholesky(corr)
                L = np.tril(L.cpu().detach().numpy())
                L = L[L != 0]
            else:
                L = [[1]]
            
            self.L = torch.tensor(L).float()
            self.beta = self.model_start.beta
            self.b = self.model_start.b
            self.phi = self.model_start.phi
            self.lambd1 = self.model_start.lambd1
            self.lambd2 = self.model_start.lambd2
 
        
        outs = []
        configs = {'random_seed':[], 'lr':[], 'scheduler_factor':[], 'loglike':[]}
        r=0
        for lr in tqdm(lrs, desc="Different lrs"):
            for scheduler_factor in tqdm(scheduler_factors, desc="Different scheduler factors"):
                for _ in tqdm(range(reps), desc="Reps"):
                    outs.append(fit_model(X,
                                          Y,
                                          D,
                                          C,
                                          K = self.K,
                                          L = self.L,
                                          beta = self.beta,
                                          b = self.b,
                                          phi = self.phi,
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
        self.loglike = self.outs[ind]['loglike']
        self.gradnorm = self.outs[ind]['grad_norm']
        self.aic = self.outs[ind]['aic']
        self.sigma = ch(self.outs[ind]['L'], tril=False).cpu().numpy()
        self.L = self.outs[ind]['L'].cpu().numpy()
        self.lambd = self.outs[ind]['lambd'].cpu().numpy()
        self.phi = self.outs[ind]['phi'].cpu().numpy()
        self.b = self.outs[ind]['b'].cpu().numpy()
        self.beta = self.outs[ind]['beta'].cpu().numpy()

    def sample_alpha(self,
                     X,
                     Y,
                     D,
                     C,
                     fam_number,
                     n_samples=1000,
                     burn_in=100,
                     thinning=20,
                     return_map=False,
                     random_seed = 42):
        
        return SampleAlpha(0,
                           betas=[self.beta],
                           lambds=[self.lambd],
                           bs=[self.b],
                           phis=[self.phi],
                           sigmas=[self.sigma],
                           D=D,
                           fam_number=fam_number,
                           X=X,
                           Y=Y,
                           C=C,
                           K=self.K,
                           random_seed=random_seed,
                           n_samples=n_samples,
                           burn_in=burn_in,
                           thinning=thinning,
                           return_map=return_map).numpy().reshape((-1,self.K))  
    
    def predict(self,
                X_train, Y_train, D_train,
                X_test, D_test,
                C):

        sig = lambda x: 1/(1+np.exp(-x))
        fam_numbers = np.argmax(D_test, axis=1)
        A_hash = {}
        for fam_number in np.unique(fam_numbers):
            A_hash[fam_number] = self.sample_alpha(X_train,
                                                   Y_train,
                                                   D_train,
                                                   C,
                                                   fam_number,
                                                   return_map=True)
        A_map = np.vstack([A_hash[fam_number] for fam_number in fam_numbers])
        mu = C+(1-C)*sig((X_test@self.beta+A_map)@self.lambd+self.b)
        return mu
        
def log_like_fe(X, Y, D, C, A, beta, lambd, b, phi):
    mu = C+(1-C)*sigmoid(torch.clamp(((X@beta+D@A)@lambd+b), min=-4.5, max=4.5))
    beta_dist = Beta(phi*mu, phi*(1-mu))

    nan_mask = torch.isnan(Y) #dealing with missing values
    Y_clipped = torch.where(nan_mask, torch.full_like(Y, 0.5), Y)
    ll_terms = beta_dist.log_prob(Y_clipped[None, :])
    loglike = torch.where(nan_mask, torch.zeros_like(ll_terms), ll_terms).sum()
    return loglike
    
class ScalingLawFixedEffects:
    def __init__(self, K):
        self.K = K
        self.A = None
        self.beta = None
        self.b = None
        self.phi = None
        self.lambd1 = None
        self.lambd2 = None
        
    def fit(self,
            X,
            Y,
            D,
            C,
            lr = .5,  # Initial learning rate
            reg = 1e1,
            scheduler_factor: float = .9,    # reduction factor on plateau
            scheduler_patience: int = 30,     # epochs with no improvement before reducing
            scheduler_threshold: float = 1e-3,
            n_epochs = 15000,
            scale = 1e-4,
            tol = 1e-4,
            verbose = True,
            print_every = 500,
            device='cpu',
            random_seed = 42):
        
        N = D.shape[1]
        p = X.shape[1]
        J = Y.shape[1]
        K = self.K
            
        # Initializing parameters for training
        torch.manual_seed(random_seed)
        if self.A is None: self.A = Parameter(torch.normal(0, scale, size=(N,K), device=device))
        if self.beta is None: self.beta = Parameter(torch.normal(0, scale, size=(p, K), device=device))
        if self.b is None: self.b = Parameter(torch.normal(0, scale, size=(1, J), device=device))
        if self.phi is None: self.phi = Parameter(torch.normal(0, scale, size=(1, J), device=device).abs())
        if self.lambd1 is None: self.lambd1 = Parameter(torch.normal(0, scale, size=(K,), device=device).abs())
        if self.lambd2 is None: self.lambd2 = Parameter(torch.normal(0, scale, size=(K,J-K), device=device))

          
        # Initialize parameters
        parameters = [self.A, self.lambd1, self.lambd2, self.phi, self.b, self.beta]
        optimizer = Adam(parameters, lr=lr)
        scheduler  = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            threshold=scheduler_threshold,
            threshold_mode='abs'
        )
    
        hist = []
        
        for ep in tqdm(range(n_epochs), desc="Training FE model", disable=not verbose):
            
            lambd = torch.hstack((torch.diag(self.lambd1), self.lambd2)) 
            phi = self.phi
            
            # Zero gradients
            for param in parameters:
                if param.grad is not None:
                    param.grad.zero_()
            
            # Compute the initial loss
            loss = - log_like_fe(X, Y, D, C, self.A, self.beta, lambd, self.b, phi)
            loss /= Y.shape[0] * Y.shape[1]
            if self.K>1: loss += reg * torch.square(torch.diag(torch.cov(self.A.T)) - 1).sum()
            else: loss += reg * torch.square(torch.var(self.A) - 1).sum()
            hist.append(loss.item())

            # Compute gradients
            loss.backward()
                
            # Calculate the gradient norm
            grad_norm = 0
            for param in parameters:
                grad_norm += (param.grad**2).sum()
            grad_norm = (grad_norm)**.5

            # Print diagnostics
            if verbose: 
                if (ep+1)%print_every==0:
                    print(f"epoch = {ep+1:10}, Grad {grad_norm.item():.10f}, Loss {loss.item():.10f}, LR {scheduler.get_last_lr()[0]:.10f}")
            if grad_norm<tol:
                print(f"epoch = {ep+1:10}, Grad {grad_norm.item():.10f}, Loss {loss.item():.10f}, LR {scheduler.get_last_lr()[0]:.10f}")
                break
            
            # Update
            optimizer.step()
            scheduler.step(loss.item())

            # Projection step
            with torch.no_grad():
                # phi
                self.phi.clamp_(min=0, max=None)
    
                # Lambd
                self.lambd1.clamp_(min=0, max=None)
            
            self.hist = hist

        # Out
        self.A = self.A.detach().cpu()
        self.lambd1 = self.lambd1.detach().cpu()
        self.lambd2 = self.lambd2.detach().cpu()
        self.phi = self.phi.detach().cpu()
        self.b = self.b.detach().cpu()
        self.beta = self.beta.detach().cpu()


def joint_dist_one_fam(X, Y, C, A, beta, lambd, b, phi, sigma, device='cpu'):
    D = torch.eye(1).to(device)
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(A.squeeze().shape[0], device=device),
        covariance_matrix=sigma.to(device)
    )
    return log_like_fe(X.to(device), Y.to(device), D.to(device), C.to(device), A.to(device).squeeze()[None,:], beta.to(device), lambd.to(device), b.to(device), phi.to(device)) + prior.log_prob(A.to(device).squeeze()).to(device)


def metropolis_hastings(init,
                        sigma_prop,
                        X_tensor,
                        Y_tensor,
                        C_tensor,
                        beta_tensor,
                        lambd_tensor,
                        b_tensor,
                        phi_tensor,
                        sigma_tensor,
                        n_samples=2000,
                        burn_in=100,
                        thinning=10):
    """
    Perform Metropolis-Hastings sampling in the log domain using PyTorch with thinning.
    
    Parameters:
    - init: torch.Tensor, initial state of the chain.
    - sigma: torch.Tensor, covariance matrix for the Gaussian proposal distribution.
    - n_samples: int, number of samples to draw after burn-in.
    - burn_in: int, number of initial samples to discard.
    - thinning: int, interval at which to keep samples (e.g., 2 keeps every second sample).
    
    Returns:
    - samples: torch.Tensor of shape (n_samples, d), sampled points.
    """
    def log_density(A):
        return joint_dist_one_fam(X_tensor, Y_tensor, C_tensor, 
                                  A, beta_tensor, lambd_tensor, 
                                  b_tensor, phi_tensor, sigma_tensor)
    
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

    
### Computing Asympt Variance
def pack_theta(beta, lambd1, lambd2, b, phi, sigma_lt):
    return torch.cat([
        beta.contiguous().view(-1),
        lambd1.contiguous().view(-1),
        lambd2.contiguous().view(-1),
        b.contiguous().view(-1),
        phi.contiguous().view(-1),
        sigma_lt.contiguous().view(-1)
    ])

def unpack_theta(theta, shapes):
    # shapes is a tuple: (beta_shape, lambd1_shape, lambd2_shape, b_shape, phi_shape, sigma_lt_shape)
    beta_shape, lambd1_shape, lambd2_shape, b_shape, phi_shape, sigma_lt_shape = shapes
    sizes = [torch.tensor(s).prod().item() for s in shapes]
    
    idx1 = sizes[0]
    idx2 = idx1 + sizes[1]
    idx3 = idx2 + sizes[2]
    idx4 = idx3 + sizes[3]
    idx5 = idx4 + sizes[4]
    
    beta_new = theta[:idx1].view(beta_shape)
    lambd1_new = theta[idx1:idx2].view(lambd1_shape)
    lambd2_new = theta[idx2:idx3].view(lambd2_shape)
    b_new      = theta[idx3:idx4].view(b_shape)
    phi_new    = theta[idx4:idx5].view(phi_shape)
    sigma_lt_new = theta[idx5:].view(sigma_lt_shape)
    
    return beta_new, lambd1_new, lambd2_new, b_new, phi_new, sigma_lt_new

def extract_lower_triangle(corr_matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts the lower triangular part (excluding the diagonal) from a correlation matrix.
    
    Parameters:
        corr_matrix (torch.Tensor): A square symmetric correlation matrix.
    
    Returns:
        torch.Tensor: A 1D tensor containing the elements from the lower triangle (without the diagonal).
    """
    n, m = corr_matrix.shape
    if n != m:
        raise ValueError("Input matrix must be square.")
    # Get indices for the lower triangle, excluding the diagonal.
    tril_indices = torch.tril_indices(n, n, offset=-1)
    return corr_matrix[tril_indices[0], tril_indices[1]]

def reconstruct_corr_matrix(lower_triangle: torch.Tensor) -> torch.Tensor:
    """
    Reconstructs a full symmetric correlation matrix from its vectorized lower triangular part (excluding the diagonal).
    The diagonal of the correlation matrix is set to 1.
    
    The length of `lower_triangle` should be n*(n-1)/2 for some integer n.
    
    Parameters:
        lower_triangle (torch.Tensor): A 1D tensor containing the lower triangular elements (excluding the diagonal)
                                       of a correlation matrix.
    
    Returns:
        torch.Tensor: The reconstructed symmetric correlation matrix with ones on the diagonal.
                      The result remains on the same device as the input and supports gradient propagation.
    """
    m = lower_triangle.numel()
    # Solve n*(n-1)/2 = m for n.
    n_float = (1 + torch.sqrt(torch.tensor(1 + 8 * m, 
                                             dtype=lower_triangle.dtype, 
                                             device=lower_triangle.device))) / 2
    n = int(n_float.item())
    if n * (n - 1) // 2 != m:
        raise ValueError("The length of lower_triangle does not correspond to a valid square matrix.")
    
    # Create a flat tensor of zeros on the same device and type.
    flat_output = torch.zeros(n * n, dtype=lower_triangle.dtype, device=lower_triangle.device)
    
    # Get the row and column indices for the lower triangle (excluding the diagonal).
    i, j = torch.tril_indices(n, n, offset=-1, device=lower_triangle.device)
    flat_indices = i * n + j  # convert 2D indices into flat indices
    
    # Use the differentiable scatter operation to place lower_triangle values in the flat tensor.
    flat_output = flat_output.scatter(dim=0, index=flat_indices, src=lower_triangle)
    
    # Reshape to an n x n matrix; this matrix now contains the lower-triangular values in their proper positions.
    M_lower = flat_output.view(n, n)
    
    # Construct the full symmetric matrix by adding its transpose and setting the diagonal to 1.
    # The final matrix is: lower triangle + (lower triangle)^T + I.
    corr_matrix = M_lower + M_lower.t() + torch.eye(n, dtype=lower_triangle.dtype, device=lower_triangle.device)
    return corr_matrix

def GetAsymptVar(X, Y, D, C, model, K,
                 device = 'cuda',
                 B = 25000,
                 lr = 1e-4,
                 grad_tol = 1e-3,
                 n_epochs = 100000,
                 random_seed=42):
    
    norm_dist = MultivariateNormal(torch.zeros(K), torch.eye(K))
    torch.manual_seed(random_seed)
    Z = norm_dist.sample(sample_shape=(B,)).to(device).double()
    beta = torch.tensor(model.beta,
                        device=device,
                        dtype=torch.float64,
                        requires_grad=True)
    lambd1 = torch.tensor(np.diag(model.lambd[:,:K]),
                          device=device,
                          dtype=torch.float64,
                          requires_grad=True)
    lambd2 = torch.tensor(model.lambd[:,K:],
                          device=device,
                          dtype=torch.float64,
                          requires_grad=True)
    b = torch.tensor(model.b,
                     device=device,
                     dtype=torch.float64,
                     requires_grad=True)
    phi = torch.tensor(model.phi,
                       device=device,
                       dtype=torch.float64,
                       requires_grad=True)
    sigma_lt = torch.tensor(extract_lower_triangle(torch.tensor(model.sigma)).clone().detach().numpy(),
                            device=device,
                            dtype=torch.float64,
                            requires_grad=True)
    param_shapes = (beta.shape, lambd1.shape, lambd2.shape, b.shape, phi.shape, sigma_lt.shape)

    def loss_from_theta(theta, Z=Z, jac=False, device=device):
        beta_new, lambd1_new, lambd2_new, b_new, phi_new, sigma_lt_new = unpack_theta(theta, param_shapes)
        lt = torch.linalg.cholesky(reconstruct_corr_matrix(sigma_lt_new))
        mask = torch.tril(torch.ones_like(lt)).bool()
        L_new = -lt[mask]
        lambd_new = torch.hstack((torch.diag(lambd1_new), lambd2_new))
        loss = log_like(torch.tensor(X).to(device).double(),
                         torch.tensor(Y).to(device).double(),
                         torch.tensor(D).to(device).double(),
                         torch.tensor(C).to(device).double(),
                         beta_new,
                         lambd_new,
                         b_new,
                         phi_new,
                         L_new,
                         K,
                         Z,
                         agg=not jac)
        if jac:
            return loss
        else:
            return -loss

    ## Fine-tuning
    parameters = [sigma_lt, lambd1, lambd2, phi, b, beta]
    optimizer = Adam([
        {'params': sigma_lt, 'lr': lr},
        {'params': lambd1,    'lr': lr},
        {'params': lambd2,    'lr': lr},
        {'params': phi,       'lr': 10*lr},
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
        theta = pack_theta(beta, lambd1, lambd2, b, phi, sigma_lt)
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
    phi = phi.cpu()
    sigma_lt = sigma_lt.cpu()
    Z = Z.cpu()
    theta = pack_theta(beta, lambd1, lambd2, b, phi, sigma_lt)
    def loss_from_theta2(theta):
        return loss_from_theta(theta, Z=Z, device=device)
    def loss_from_theta3(theta):
        return loss_from_theta(theta, Z=Z, jac=True, device=device)
        
    H = hessian(loss_from_theta2, theta)
    H = (H+H.T)/2
    J = jacobian(loss_from_theta3, theta)

    V = torch.linalg.inv(H)
    V2 = torch.linalg.inv((J.T@J))
    V3 = torch.linalg.inv((H + J.T@J)/2)

    ## Output
    return [[{'min_eigenvalue':np.min(torch.linalg.eig(V).eigenvalues.cpu().numpy().astype(float)),'H':H.detach().cpu().numpy(),'V':V.detach().cpu().numpy()},{name+"_ste":np.sqrt(x.detach().cpu().numpy()) for x,name in zip(unpack_theta(torch.diag(V), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi', 'sigma_lt'])}],
 [{'min_eigenvalue':np.min(torch.linalg.eig(V2).eigenvalues.cpu().numpy().astype(float)),'V':V2.detach().cpu().numpy()},{name+"_ste":np.sqrt(x.detach().cpu().numpy()) for x,name in zip(unpack_theta(torch.diag(V2), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi', 'sigma_lt'])}],
 [{'min_eigenvalue':np.min(torch.linalg.eig(V3).eigenvalues.cpu().numpy().astype(float)),'V':V3.detach().cpu().numpy()},{name+"_ste":np.sqrt(x.detach().cpu().numpy()) for x,name in zip(unpack_theta(torch.diag(V3), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi', 'sigma_lt'])}]]
    
def SampleParams(beta, b, phi, lambd1, lambd2, sigma, cov, n=100):
    sigma_lt = extract_lower_triangle(sigma)
    mu = pack_theta(torch.tensor(beta), torch.tensor(lambd1), torch.tensor(lambd2), torch.tensor(b), torch.tensor(phi), torch.tensor(sigma_lt)).numpy()
    param_shapes = (beta.shape, lambd1.shape, lambd2.shape, b.shape, phi.shape, sigma_lt.shape)
    thetas = np.random.multivariate_normal(mu, cov, size=n)
    thetas = [{name:x.detach().cpu().numpy() for x,name in zip(unpack_theta(torch.tensor(t), param_shapes), ['beta', 'lambd1', 'lambd2', 'b', 'phi', 'sigma_lt'])} for t in thetas]
    
    betas=[]
    bs=[]
    phis=[]
    lambds=[]
    sigmas=[]
    for i in range(len(thetas)):
        beta = thetas[i]['beta']
        b = thetas[i]['b']
        phi = thetas[i]['phi']
        lambd = np.hstack((np.diag(thetas[i]['lambd1']), thetas[i]['lambd2']))
        sigma = reconstruct_corr_matrix(torch.tensor(thetas[i]['sigma_lt'])).numpy()
    
        V,U=np.linalg.eigh(sigma)
        sigma = U@np.diag([v if v > 0 else 0 for v in V])@U.T #projection on PSD
        eps = 1e-4
        sigma += eps*np.eye(sigma.shape[0]) #regularization / non-degenerate distribution

        phi[phi<eps] = eps #projection on positive
        
        betas.append(beta)
        bs.append(b)
        phis.append(phi)
        lambds.append(lambd)
        sigmas.append(sigma)

    return betas, bs, phis, lambds, sigmas
    
def SampleAlpha(i,
                 betas,
                 lambds,
                 bs,
                 phis,
                 sigmas,
                 D,
                 fam_number,
                 X,
                 Y,
                 C,
                 K,
                 random_seed=42,
                 n_samples=10,
                 burn_in=100,
                 thinning=30,
                 return_map=False):
    
    torch.manual_seed(random_seed)
    
    # Select family
    ind_fam = D[:, fam_number] == 1
    X_tensor = torch.tensor(X, requires_grad=False)[ind_fam].float()
    Y_tensor = torch.tensor(Y, requires_grad=False)[ind_fam].float()
    C_tensor = torch.tensor(C, requires_grad=False).float()
    beta_tensor = torch.tensor(betas[i], requires_grad=False).float()
    lambd_tensor = torch.tensor(lambds[i], requires_grad=False).float()
    b_tensor = torch.tensor(bs[i], requires_grad=False).float()
    phi_tensor = torch.tensor(phis[i], requires_grad=False).float()
    sigma_tensor = torch.tensor(sigmas[i], requires_grad=False).float()
    
    # Optimize A_map
    A_map = torch.zeros(K, requires_grad=True).float()
    def loss_function(A_var):
        return -joint_dist_one_fam(X_tensor, Y_tensor, C_tensor, 
                                   A_var, beta_tensor, lambd_tensor, 
                                   b_tensor, phi_tensor, sigma_tensor)
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
        A_map, A_sigma, X_tensor, Y_tensor, C_tensor,
        beta_tensor, lambd_tensor, b_tensor, phi_tensor, sigma_tensor,
        n_samples=n_samples, burn_in=burn_in, thinning=thinning
    ).cpu().numpy()