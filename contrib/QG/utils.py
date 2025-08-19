import torch
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
import xarray as xr
import src.data
import xarray as xr

def load_qg_data(path, obs_from_tgt=False):
    ds = (
        xr.open_dataset(path)
        .load()
        .assign(
            input=lambda ds: ds['sf_obs'],
            tgt=lambda ds: ds['stream_function']
        )
    )
    
    return (
        ds[[*src.data.TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )

def generate_correlated_fields(batch_size, num_fields, N, L, T_corr, sigma, device='cpu', seed=41):
    """
    Generate a batch of 2D fields with spatial and temporal correlations.

    Parameters:
        batch_size (int): Number of independent realizations.
        num_fields (int): Number of fields per batch.
        N (int): Grid size (assumed to be square, NxN).
        L (float): Spatial correlation length.
        T_corr (float): Temporal correlation length.
        sigma (float): Standard deviation of the field.
        device (str): Device to run computations ('cpu' or 'cuda').
        seed (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Shape (batch_size, num_fields, N, N).
    """
    # if seed is None:
    #     seed = 41
    # torch.manual_seed(int(seed))

    # Define time points and temporal correlation matrix
    time_points = torch.linspace(0, num_fields - 1, num_fields, device=device)
    C_temporal = torch.exp(-abs(time_points[:, None] - time_points[None, :]) / T_corr)
    L_chol = torch.linalg.cholesky(C_temporal)  # Cholesky decomposition

    # Generate white noise: (batch_size, num_fields, N, N)
    white_noises = torch.randn((batch_size, num_fields, N, N), device=device)

    # Apply temporal correlation using matrix multiplication
    temporal_correlated_noises = torch.matmul(L_chol, white_noises.view(batch_size, num_fields, -1)).view(batch_size, num_fields, N, N)

    # Generate 2D wavenumber grid for spatial correlation
    kx = torch.fft.fftfreq(N, device=device) * N
    ky = torch.fft.fftfreq(N, device=device) * N
    k = torch.sqrt(kx[:, None]**2 + ky[None, :]**2)
    cutoff_mask = (k < 20).float()  # Apply high-frequency cutoff
    P_k = torch.exp(-0.5 * (k * L)**3)
    P_k[0, 0] = 0.0  # Remove DC component
    P_k = P_k / torch.sum(P_k)  # Normalize

    # Generate spatially correlated fields
    fields = []
    for i in range(num_fields):
        noise_ft = torch.fft.fft2(temporal_correlated_noises[:, i])  # FFT per batch
        field_ft = noise_ft * torch.sqrt(P_k) * cutoff_mask  # Apply correlation
        field = torch.fft.ifft2(field_ft).real  # Inverse FFT
        field = sigma * (field - field.mean(dim=(1, 2), keepdim=True)) / field.std(dim=(1, 2), keepdim=True)
        fields.append(field)

    return torch.stack(fields, dim=1)  # Shape: (batch_size, num_fields, N, N)


# We now perturb the fields for given dispalceemnt fields dx and dy
def warp_field(field, dx, dy):
    """
    Warp a 2D field based on displacement fields dx and dy.
    field (torch.Tensor): Input field of shape (num_fields, channels, height, width)
    dx (torch.Tensor): X-displacement field of shape (num_fields, height, width)
    dy (torch.Tensor): Y-displacement field of shape (num_fields, height, width)
    """
    num_fields, _, height, width = field.shape
    
    # Create base grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    base_grid = torch.stack((x, y), dim=-1).float()
    dx=dx.to(field.dtype) 
    dy=dy.to(field.dtype) 
    # Add batch dimension and move to the same device as input field
    base_grid = base_grid.unsqueeze(0).repeat(num_fields,1,1,1).to(field.device) # shape (num_fields, height, width, 2)

    # Apply displacements
    sample_grid = base_grid + torch.stack((dx, dy), dim=-1)
    sample_grid[..., 0] = sample_grid[..., 0] % (width)
    sample_grid[..., 1] = sample_grid[..., 1] % (height)
    

    # Normalize grid to [-1, 1] range
    sample_grid[..., 0] = 2 * sample_grid[..., 0] / (width) - 1
    sample_grid[..., 1] = 2 * sample_grid[..., 1] / (height) - 1

    # Perform sampling
    warped_field = F.grid_sample(field, sample_grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return warped_field.squeeze(1) # shape (num_fields, height, width)

def warp_field_batch(field, L, T_corr, sigma, init_type='perturbed', seed1=None, seed2=None):
    """
    Warp a batch of 2D fields using correlated displacement fields.

    Parameters:
        field (torch.Tensor): Input field of shape (batch_size, num_fields, height, width).
        L (float): Spatial correlation length for displacement fields.
        T_corr (float): Temporal correlation length for displacement fields.
        sigma (float): Standard deviation for displacement fields.
        device (str): Device to run computations ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Warped field of shape (batch_size, num_fields, height, width).
    """
    
    batch_size, num_fields, height, width = field.shape
    device = field.device
    if init_type=='no-perturbed':
        return field

    #seed1=# default seeds 41
    #seed2=39 # default seeds 39
    if seed1 is None and seed2 is None:
        # Generate random seeds for reproducibility
        seed1 = torch.randint(0, 2**32 - 1, (1,)).item()
        seed2 = torch.randint(0, 2**32 - 1, (1,)).item()
        # Use a fixed seed for reproducibility
    # Generate correlated displacement fields (dx, dy) with the same shape as field
    dx = generate_correlated_fields(batch_size, num_fields, height, L, T_corr, sigma, device,seed=seed1)
    dy = generate_correlated_fields(batch_size, num_fields, height, L, T_corr, sigma, device,seed=seed2)

    # Create base grid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    base_grid = torch.stack((x, y), dim=-1).float()
    dx=dx.to(field.dtype) 
    dy=dy.to(field.dtype) 
    
    # Add batch dimension and move to the same device as input field
    base_grid = base_grid.unsqueeze(0).repeat(num_fields,1,1,1).to(field.device)
    warped_fields=[]
    # create a loop and perform batching operation for now:
    for i in range(batch_size):
        field_=warp_field(field[i].unsqueeze(1), dx[i], dy[i])
        warped_fields.append(field_)

    return torch.stack(warped_fields, dim=0)