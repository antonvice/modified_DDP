import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DDP(nn.Module):
  
  '''The DDP class is a PyTorch module that implements a diffusion process for image generation. 
  The class takes four parameters: input_dim, which specifies the number of input features; output_dim, 
  which specifies the number of output features; num_diffusion_steps, which specifies the number of steps in the diffusion process; 
  and hidden_dim, which specifies the number of hidden units in the diffusion step networks.'''
  
    def __init__(self, input_dim, output_dim, num_diffusion_steps, hidden_dim=256):
      
      '''The constructor method __init__ initializes the class parameters 
      and defines the diffusion step and reverse diffusion step networks using the nn.Sequential module. 
      Each network consists of several layers, including linear layers, batch normalization layers, LeakyReLU activation functions, and dropout layers. 
      The diffusion step network outputs the generated image, 
      while the reverse diffusion step network reconstructs the input image from the generated image.'''
      
        super(DDP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.hidden_dim = hidden_dim
        
        # Define the diffusion step network
        self.diffusion_step = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Define the reverse diffusion step network
        self.reverse_diffusion_step = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
      
      '''The forward method applies the diffusion process to the input image x. 
      In each iteration of the diffusion process, the generated image x_t is computed using the diffusion step network, 
      and noise is added to x_t. Then, the reverse diffusion step network reconstructs the input image x from x_t. 
      The final reconstructed image is returned as output.'''
      
        # Apply the diffusion process
        for i in range(self.num_diffusion_steps):
            x_t = self.diffusion_step(x)
            noise = torch.randn_like(x_t)
            x_t = x_t + noise
            x = self.reverse_diffusion_step(x_t)
        return x
