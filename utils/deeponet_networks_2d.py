import numpy as np
import torch
from torch import nn
import math
import seaborn as sns 
sns.set()

import warnings
warnings.filterwarnings("ignore")

class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

#========================== Vanilla_NO_model =========================#
class Vanilla_NO_model(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs):
        """
        bs    :  Batch size.
        c     :  Number of channels in input field image.
        h     :  Number of pixels along the height of input field image.
        w     :  Number of pixels along the width of input field image.
        neval :  Number of points at which output field is evaluated for a given random input = nt*nx*ny
        p     :  Number of output neurons in both branch and trunk net.   
        
        branch inputs shape: (bs, c, h, w) 
        trunk inputs shape : (neval, 3) # 3 corresponds to t, x and y
        
        shapes:  inputs shape         -->      outputs shape
        branch:  (bs, c, h, w)        -->      (bs, p)
        trunk:   (neval, 3)           -->      (neval, p)
        
        outputs shape: (bs, neval).
        """
        
        branch_outputs = self.branch_net(branch_inputs)
        trunk_outputs = self.trunk_net(trunk_inputs)
        
        results = torch.einsum('ik, lk -> il', branch_outputs, trunk_outputs)
        
        return results


#========================== Latent_NO_model =========================#
class Latent_DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net, latent_dim):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.latent_dim = latent_dim
    
    def forward(self, branch_inputs, trunk_inputs):
        """
        bs      :  Batch size.
        c       :  Number of channels in input field image.
        h       :  Number of pixels along the height of input field image.
        w       :  Number of pixels along the width of input field image.
        neval_t :  Number of time points at which latent output field is evaluated for a given random input field sample = nt
        p       :  Number of output neurons in both branch and trunk net. 
        d_z     :  Number of sensors on each latent field.   
        
        branch inputs shape: (bs, c, h, w) 
        trunk inputs shape : (neval_t, 1) # 1 corresponds to t
        
        shapes:  inputs shape         -->      outputs shape
        branch:  (bs, c, h, w)        -->      (bs, p)
        trunk:   (neval_t, 1)         -->      (neval_t, p)
        
        reshape p to (d_z, p_)
        
        outputs shape: (bs, neval_t, d_z).
        """
        
        branch_outputs = self.branch_net(branch_inputs)
        trunk_outputs = self.trunk_net(trunk_inputs)
        
        branch_outputs_ = branch_outputs.reshape(branch_inputs.shape[0], self.latent_dim, -1) # (bs, d_z, p_)
        trunk_outputs_ = trunk_outputs.reshape(trunk_inputs.shape[0], self.latent_dim, -1) # (neval_t, d_z, p_)
        
        results = torch.einsum('ijk, ljk -> ilj', branch_outputs_, trunk_outputs_)
        
        return results

class Reconstruction_DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs):
        """
        bs        :  Batch size.
        neval_t   :  Number of time points at which latent output field is evaluated for a given random input field sample = nt
        d_z       :  Number of sensors on each latent field. 
        neval_loc :  Number of locations at which output field is evaluated at each time point = nx*ny
        q         :  Number of output neurons in both branch and trunk net.   
        
        branch inputs shape: (bs, neval_t, d_z) 
        trunk inputs shape : (neval_loc, 2) # 2 corresponds to x and y
        
        shapes:  inputs shape            -->      outputs shape
        branch:  (bs, neval_t, d_z)      -->      (bs, neval_t, q)
        trunk:   (neval_loc, 2)          -->      (neval_loc, q)
        
        outputs shape: (bs, neval_t, neval_loc).
        """
        
        branch_outputs = self.branch_net(branch_inputs) # (bs, neval_t, q)
        trunk_outputs = self.trunk_net(trunk_inputs) # (neval_loc, q)

        results = torch.einsum('ijk, lk -> ijl', branch_outputs, trunk_outputs)
        
        return results

class Latent_NO_model(nn.Module):
    def __init__(self, latent_branch_net, latent_trunk_net, latent_dim, reconstruction_branch_net, reconstruction_trunk_net):
        super(Latent_NO_model, self).__init__()
        self.latent_model = Latent_DeepONet(latent_branch_net, latent_trunk_net, latent_dim)
        self.reconstruction_model = Reconstruction_DeepONet(reconstruction_branch_net, reconstruction_trunk_net)
    
    def forward(self, latent_branch_inputs, latent_trunk_inputs, reconstruction_trunk_inputs):
        latent_out = self.latent_model(latent_branch_inputs, latent_trunk_inputs)
        reconstruction_out = self.reconstruction_model(latent_out, reconstruction_trunk_inputs)
        return latent_out, reconstruction_out