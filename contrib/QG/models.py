import torch
import torch.nn.functional as F
import torch.nn as nn
from src.models import Lit4dVarNet, GradSolver  # Import the base class for function overloading


class Litvarcost4dVarNet(Lit4dVarNet):
    def __init__(self, solver, rec_weight, opt_fn, dynamical_model_solver, test_metrics=None, pre_metric_fn=None, norm_stats=None, persist_rw=True):
        # Call the parent class's initializer
        super().__init__(solver, rec_weight, opt_fn, test_metrics, pre_metric_fn, norm_stats, persist_rw)
        
        # Add your custom attributes
        self.dynamical_model_solver = dynamical_model_solver
    def base_step(self, batch, phase=""):
        out = self(batch=batch)

        # compute the number of time points inside the out to subtract 1.
        with torch.no_grad():
            dynamical_target = self.dynamical_model_solver(out[0:8])

        loss = self.weighted_mse(out[1:] - dynamical_target, self.rec_weight)

        with torch.no_grad():
            self.log(f"{phase}_mse", 10000 * loss * self.norm_stats[1]**2, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f"{phase}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss, out

    
class ICGradSolver(GradSolver):
    def __init__(self, prior_cost, obs_cost, grad_mod, init_state_fn, n_step, lr_grad=0.2, **kwargs):
        # Call the parent class's initializer
        super().__init__(prior_cost, obs_cost, grad_mod, n_step, lr_grad, **kwargs)
        
        # Store the additional function
        self.init_state_fn = init_state_fn

    def init_state(self, batch, x_init=None):
        """
        Overwrite the init_state method to include the additional function logic.
        """
        # Use the provided initial condition during inference
        if not self.training and x_init is not None:
            return x_init.detach().requires_grad_(True)
        else:
        # Use the perturbation-based initialization during training
            x_init = self.init_state_fn(batch.tgt)
        x_init = x_init.detach().requires_grad_(True)
        return x_init

# Implement a model that will upsample, and the compute the dynamical loss


# Implement 
