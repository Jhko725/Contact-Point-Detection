from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt

class GeneralModeDataset(Dataset):
    """
    A PyTorch Dataset to handle general mode AFM data. 
    """

    def __init__(self, t, d_list, x0, z_list, ode_params):
        """
        Parameters
        ----------
        t : 1D Numpy array 
            1D Numpy array containing the time stamps corresponding to the ODE solution x(t)
        ode_params : dict
            Dictionary containing the necessary parameters for the ODE. 
            Required form is ode_params = {'A0' : float, 'Q' : float, 'Om' : float, 'k' : float}
        """
        # Needs modifying - in the final form, we do not necessarily need x0 in both the model and the dataset
        self.t = np.array(t)
        self.d_list = d_list
        self.z_list = z_list
        self.ode_params = ode_params
        self.x0 = x0

    def __len__(self):
        return len(self.d_list)

    def __getitem__(self, idx):
        sample = {'time': self.t, 'd': self.d_list[idx], 'x0': self.x0, 'z': self.z_list[idx][:]}
        return sample

    def __eq__(self, other):
        """
        Comparison between two GeneralModeDataset objects. True if both objects have the same ODE parameters.
        """
        return self.ode_params == other.ode_params

    def PlotData(self, idx, figsize = (7, 5), fontsize = 14):
        data = self.__getitem__(idx)
        
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        ax.plot(data['time'], data['z'], '.k')
        ax.grid(ls = '--')
        ax.set_xlabel('Normalized Time', fontsize = fontsize)
        ax.set_ylabel('z (nm)', fontsize = fontsize)

        return fig, ax

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
        super(F_cons, self).__init__()
        self.hidden_nodes = list(hidden_nodes)
        self.layers = nn.ModuleList()
        for i in range(len(self.hidden_nodes)):
            if i == 0:
                self.layers.append(nn.Linear(1, self.hidden_nodes[i]))
            else:
                self.layers.append(nn.Linear(self.hidden_nodes[i-1], self.hidden_nodes[i]))
        self.layers.append(nn.Linear(self.hidden_nodes[-1], 1))

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        # Initialize last layer weight and bias to zero tensors
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, z):
        """
        Returns the neural network output as a function of input z.
        z is assumed to be a tensor with size [batch_size, 1].

        Parameters
        ----------
        z : tensor with dimensions [batch_size, 1].
            Neural network input. Represents the instantaneous tip-sample distance.

        Returns
        -------
        F : tensor with dimensions [batch_size, 1].
            Neural network output. Represents the modeled tip-sample force.
        """
        interm = self.layers[0](z)
        
        for layer in self.layers[1:]:
            interm = self.elu(interm)
            interm = layer(interm)

        F = self.tanh(interm)
        return F

class AFM_NeuralODE(nn.Module):
    """
    A  Pytorch module to create a NeuralODE modeling the AFM tip-sample dynamics.
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

    def __init__(self, A0, Om, Q, k, hidden_nodes = [4]):
        """
        Parameters
        ----------
        hidden_nodes : list of int
            List of the nodes in each of the hidden layers of the model.
        A0 : float
            Free amplitude of the tip at resonance in units of [nm].
            Follows the definition outlined in M. Lee et. al., Phys. Rev. Lett. 97, 036104 (2006).
        Om : float
            Ratio between the resonance frequency f0 and the driving frequency f. Om = f/f0
        Q : float
            Q-factor of the cantilever/QTF.
        k : float
            Spring constant of the cantilever/QTF in units of [N/m].
        d : PyTorch tensor with dimensions [batch_size, 1]
            Batched mean tip-sample distance in units of [nm]. Initialized to None.
        """
        super(AFM_NeuralODE, self).__init__()
        self.Fc = F_cons(hidden_nodes)
        self.nfe = 0

        self.d = None
        self.A0 = A0
        self.Om = Om
        self.Q = Q
        self.k = k

        # Constant tensors to be used in the model
        self.C1 = torch.tensor([[-1./self.Q, -1.], [1., 0.]], device = torch.device("cuda")).unsqueeze(0) # Has size = (1, 2, 2)
        self.C2 = torch.tensor([[1.],[0.]], device = torch.device("cuda")).unsqueeze(0) # Has size = (1, 2, 1)
        self.register_buffer('Constant 1', self.C1)
        self.register_buffer('Constant 2', self.C2)

    def forward(self, t, x):
        """
        Returns the right-hand-side of the differential equation dx/dt = f(t, x)
        x is a PyTorch tensor with size [batch_size, 2], where the second dimension corresponds to length 2 vector of the x = [y, z], where y = dz/dt 

        Parameters
        ----------
        t : float
            Time
        x : PyTorch tensor with dimensions [batch_size, 2]
            The second dimension correspond to x = [y, z], where y = dz/dt

        Returns
        -------
        dxdt : PyTorch tensor with dimensions [batch_size, 2]
            The second dimension corresponds to dxdt = [dy/dt, dz/dt]
        """
        self.nfe += 1
        F = self.Fc(x[:, 1:])
    
        # The first term is broadcasted matrix multiplication of (1, 2, 2) * (b, 2, 1) = (b, 2, 1), where b = self.batch_size.
        # The second term is broadcasted matrix multiplication of (1, 2, 1) * (b, 1, 1) = (b, 2, 1)
        ode = torch.matmul(self.C1, x.unsqueeze(-1)) + torch.matmul(self.C2, (self.d + (self.A0/self.Q)*torch.cos(self.Om*t) + F/self.k).unsqueeze(-1))

        # Squeeze to return a tensor of shape (b, 2)
        return ode.squeeze(-1)


class LightningTrainer(pl.LightningModule):
    """
    A PyTorch-Lightning LightningModule for training the NeuralODE created by the class AFM_NeuralODE. 

    ...

    Attributes
    ----------
    ODE : An instance of class AFM_NeuralODE
        A NeuralODE model to represent the dynamics between the tip and the sample.
    hparams : An argparse.Namespace object with fields 'train_dataset', 'hidden_nodes', 'batch_size', 'lr', and 'solver'.
            Hyperparameters of the model. 
            'train_dataset' : An instance of the class GeneralModeDataset
                A PyTorch dataset of a given general mode approach data to train the NeuralODE with. 
            'hidden nodes' : list or array of integers
                List of number of nodes in the hidden layers of the model.
            'batch_size' : integer
                Batch_size of the model
            'lr' : float
                Learning rate
            'solver' : string, must be compatible with TorchDiffEq.odeint_adjoint
                Type of ODE solver to be used to solve the NeuralODE
    """

    #def __init__(self, train_dataset, lr = 0.05, hidden_nodes = [10, 10], batch_size = 1, solver = 'dopri5'):
    def __init__(self, hparams):
        """
        Parameters
        ----------
        hparams : An argparse.Namespace object with fields 'train_dataset', 'hidden_nodes', 'batch_size', 'lr', and 'solver'.
            Hyperparameters of the model. 
            'train_dataset' : An instance of the class GeneralModeDataset
                A PyTorch dataset of a given general mode approach data to train the NeuralODE with. 
            'hidden nodes' : list or array of integers
                List of number of nodes in the hidden layers of the model.
            'batch_size' : integer
                Batch_size of the model
            'lr' : float
                Learning rate
            'solver' : string, must be compatible with TorchDiffEq.odeint_adjoint
                Type of ODE solver to be used to solve the NeuralODE
        """
        super(LightningTrainer, self).__init__()
        self.hparams = hparams
        self.train_dataset = self.hparams.train_dataset
        ode_params = self.train_dataset.ode_params
        self.ODE = AFM_NeuralODE(**ode_params, hidden_nodes = self.hparams.hidden_nodes)
        self.batch_size = self.hparams.batch_size
        self.lr = self.hparams.lr
        self.solver = self.hparams.solver

    def forward(self, t, x0, d):
        self.ODE.d = d.view(self.batch_size, 1)

        x_pred = odeint(self.ODE, x0, t, method = self.solver)
        # x_pred has shape = [time, batch_size, 2]. Permute z_pred so that it has shape = [batch_size, time]
        z_pred = x_pred[:,:,1].permute(1,0)

        return z_pred

    def training_step(self, batch, batch_nb):
        t = batch['time'][0].float()
        x0 = batch['x0'].float()
        d = batch['d'].float()

        z_pred = self.forward(t, x0, d)
        z_true = batch['z'].float()

        log1pI_pred = self.LogSpectra(z_pred)
        log1pI_true = self.LogSpectra(z_true)

        loss_function = nn.MSELoss()
        loss = loss_function(log1pI_pred, log1pI_true)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)

    @staticmethod
    def LogSpectra(z):
        """
        Calculates the complex FFT spectra of the given signal z(t), then calculates the intensity. Finally, returns log1p(intensity), where log1p is preferred over log due to its numerical stability.

        Parameters
        ----------
        z : A 1D PyTorch tensor
            1D tensor of real values, corresponding to the time series z(t)

        Returns
        -------
        z_log1pI : A 1d Pytorch tensor
            1D tensor corresponding to the log1p of the Fourier intensity of z(t).
        """

        z_fft = torch.rfft(z, 1)
        #z_I = torch.sum(z_fft**2, dim = -1)
        z_log1pI = torch.norm(z_fft, p = 2, dim = -1)
        #z_log1pI = torch.log1p(z_I)

        return z_log1pI

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, 100, threshold = 0.05, threshold_mode = 'rel')
        #return [optim], [sched]
        return optim

    def predict_z(self):
        pass

    def predict_force(self, d):
        d = np.array(d)
        d_tensor = torch.from_numpy(d).cuda(non_blocking = True).float().reshape(-1, 1)
        F_pred = self.ODE.Fc(d_tensor).cpu().detach().numpy()
        return F_pred

def TrainModel(model, max_epochs = 10000, checkpoint_path = './checkpoints'):
    """
    Trains the model using PyTorch_Lightning.trainer. 

    Parameters
    ----------
    model : An instance of PyTorch_Lightning.LightningModule
        Neural network to be trained. 
    max_epochs : integer
        Maximum number of training.
    checkpoint_path : path
        Path to save the checkpointed model during training.
    """
    checkpoint_callback = pl.ModelCheckpoint(filepath = checkpoint_path, save_best_only = True, verbose = True, monitor = 'loss', mode = 'min', prefix = '')
    trainer = pl.Trainer(gpus = 1, early_stop_callback = None, checkpoint_callback = checkpoint_callback, show_progress_bar = True, max_nb_epochs = max_epochs)
    trainer.fit(model)

def LoadModel(checkpoint_path):
    """
    Loads the checkpointed model from checkpoint_path. 

    Parameters
    ----------
    checkpoint_path : path
        Path to load the checkpointed model from.

    Returns
    -------
    loaded_model : An instance of PyTorch_Lightning.LightningModule
        Loaded neural network.
    """
    return LightningTrainer.load_from_checkpoint(checkpoint_path)