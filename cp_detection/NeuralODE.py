from __future__ import print_function, division
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class F_cons(nn.Module):
    """
    A PyTorch module to model the conservative force experienced by the atomic force microscope probe.
    We assume that the force only depends on z, and model it using a simple MLP.

    ...

    Attributes
    ----------
    hidden_nodes : list of int
        List of the nodes in each of the hidden layers of the model
    layers : list of torch.nn.Linear objects
        List of layers in the model. All the layers used are fully connected layers.
    elu : torch.nn.ELU object
        Elu activation layer, used for the activations between the hidden layers.
    tanh : torch.nn.Tanh object
        Tanh activation layer, used for the actvation for the model output.

    Methods
    -------
    forward(z)
        Returns the neural network output as a function of input z.
        z is assumed to be a tensor with size [1].
    """

    def __init__(self, hidden_nodes = [4]):
        """
        Parameters
        ----------
        hidden_nodes : list of int
            List of the nodes in each of the hidden layers of the model
        """
        self.hidden_nodes = list(hidden_nodes)
        self.layers = []
        for i in range(len(self.hidden_nodes)):
            if i == 0:
                self.layers.append(nn.Linear(1, self.hidden_nodes[i]))
            else:
                self.layers.append(nn.Linear(self.hidden_nodes[i-1], self.hidden_nodes[i]))
        self.layers.append(nn.Linear(self.hidden_nodes[-1], 1))

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        """
        Returns the neural network output as a function of input z.
        z is assumed to be a tensor with size [1].

        Parameters
        ----------
        z : tensor with size [1].
            Neural network input. Represents the instantaneous tip-sample distance.

        Returns
        -------
        F : tensor with size [1].
            Neural network output. Represents the modeled tip-sample force.
        """
        interm = self.layers[0](z)
        
        for layer in self.layers[1:]:
            interm = self.elu(interm)
            interm = layer(interm)

        F = self.tanh(interm)
        return F

class AFM_NeuralODE(pl.LightningModule):
    """
    A PyTorch-Lightning LightningModule for creating and training a NeuralODE modeling the AFM tip-sample dynamics. 
    Note that all length scales involved in the model are scaled by setting 1[nm] = 1 
    and all timescales scaled to w0*1[s] = 1.
    ...

    Attributes
    ----------
    Fc : An instance of class F_cons
        A MLP model to represent the conservative force between the tip and the sample.
    nfe : int
        Number of forward evaluations. Incremented by 1 everytime forward() is evaluated.
    d : float
        Mean tip-sample distance in units of [nm].
    A0 : float
        Free amplitude of the tip at resonance in units of [nm].
        Follows the definition outlined in M. Lee et. al., Phys. Rev. Lett. 97, 036104 (2006).
    Om : float
        Ratio between the resonance frequency f0 and the driving frequency f. Om = f/f0
    Q : float
        Q-factor of the cantilever/QTF.
    k : float
        Spring constant of the cantilever/QTF in units of [N/m].

    
    Methods
    -------
    forward(t, x)
        Returns the right-hand-side of the differential equation dx/dt = f(t, x)
        x is a length 2 vector of the form x = [y, z], where y = dz/dt 
    """

    def __init__(self, d, A0, Om, Q, k, hidden_nodes = [4]):
        """
        Parameters
        ----------
        hidden_nodes : list of int
            List of the nodes in each of the hidden layers of the model.
        d : float
            Mean tip-sample distance in units of [nm].
        A0 : float
            Free amplitude of the tip at resonance in units of [nm].
            Follows the definition outlined in M. Lee et. al., Phys. Rev. Lett. 97, 036104 (2006).
        Om : float
            Ratio between the resonance frequency f0 and the driving frequency f. Om = f/f0
        Q : float
            Q-factor of the cantilever/QTF.
        k : float
            Spring constant of the cantilever/QTF in units of [N/m].
        """
        super(AFM_NeuralODE, self).__init__()
        self.Fc = F_cons(hidden_nodes)
        self.nfe = 0

        self.d = d
        self.A0 = A0
        self.Om = Om
        self.Q = Q
        self.k = k

        # Constant tensors to be used in the model
        self.C1 = torch.tensor([[-1./self.Q, -1.], [1., 0.]])
        self.C2 = torch.tensor([1.,0.])
        self.register_buffer('Constant 1', self.C1)
        self.register_buffer('Constant 2', self.C2)

    def forward(self, t, x):
        """
        Returns the right-hand-side of the differential equation dx/dt = f(t, x)
        x is a length 2 vector of the form x = [y, z], where y = dz/dt 

        Parameters
        ----------
        t : float
            Time
        x : 1D PyTorch tensor with length 2
            A length 2 vector of the form x = [y, z], where y = dz/dt

        Returns
        -------
        dxdt : 1D PyTorch tensor with length 2
            A length 2 vector corresponding to dxdt = [dy/dt, dz/dt]
        """
        self.nfe += 1
        F = self.Fc(x[1].unsqueeze(-1))

        ode = torch.matmul(self.C1, x) + (self.d + (self.A0/self.Q)*torch.cos(self.Om*t) + F/self.k) * self.C2

        return ode