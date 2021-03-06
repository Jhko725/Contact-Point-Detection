from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchdiffeq import odeint_adjoint as odeint
import matplotlib.pyplot as plt
import json, sys, os, inspect
from argparse import Namespace

class GeneralModeDataset(Dataset):
    """
    A PyTorch Dataset to handle general mode AFM data. 
    """

    def __init__(self, t, d_array, x_array, ode_params):
        """
        Parameters
        ----------
        t : 1D Numpy array 
            1D Numpy array containing the time stamps corresponding to the ODE solution x(t)
        ode_params : dict
            Dictionary containing the necessary parameters for the ODE. 
            Required form is ode_params = {'A0' : float, 'Q' : float, 'Om' : float, 'k' : float}
        """

        self.t = np.array(t)
        self.d_array = np.array(d_array)
        self.x_array = np.array(x_array)
        self.z_array = self.x_array[:,1,:]
        self.ode_params = ode_params

    def __repr__(self):
        repr_str = 'A general mode dataset with ODE parameters: ' + ', '.join('{} = {}'.format(k, v) for k, v in self.ode_params.items())
        return repr_str

    def __len__(self):
        return len(self.d_array)

    def __getitem__(self, idx):
        #sample = {'time': self.t, 'd': self.d_list[idx], 'x0': self.x0, 'z': self.z_list[idx][:]}
        sample = {'time': self.t[:1000]-self.t[0], 'd': self.d_array[idx],'z': self.z_array[idx][:1000]}
        return sample

    def __eq__(self, other):
        """
        Comparison between two GeneralModeDataset objects. True if both objects have the same ODE parameters.
        """
        return self.ode_params == other.ode_params

    def AddNoise(self, SNR, seed = None):
        """
        Adds Gaussian noise corresponding to SNR to the z_array data
        SNR is defined as $SNR = <z^2(t)>/\\sigma^2$, where $noise ~ N(0, \\sigma^2)$
        """
        np.random.seed(seed)
        sqr_avg = np.mean(self.z_array**2, axis = -1)
        var = sqr_avg/SNR

        noise = np.stack([np.random.normal(0, v, size = self.z_array.shape[-1]) for v in var], axis = 0)
        
        self.z_array += noise

    def PlotData(self, idx, ax, N_data = 0, fontsize = 14, **kwargs):
        data = self.__getitem__(idx)
        z = data['z']
        t = data['time'][-len(z):]
        ax.scatter(t[-N_data:], z[-N_data:], **kwargs)
        ax.grid(ls = '--')
        ax.set_xlabel('Normalized Time', fontsize = fontsize)
        ax.set_ylabel('z (nm)', fontsize = fontsize)

        return ax

    def save(self, savepath):
        """
        Saves the given dataset in json format.

        Parameters
        ----------
        savepath : path
            Path to save the dataset at.
        """
        savedict = self.__dict__
        for k, v in savedict.items():
            if isinstance(v, np.ndarray):
                savedict[k] = v.tolist()

        with open(savepath, 'w') as savefile:
            json.dump(savedict, savefile)
        print('Saved data to: {}'.format(savepath))

    @classmethod
    def load(cls, loadpath):
        """
        Loads the dataset from a json file.

        Parameters
        ----------
        loadpath : path
            Path to the json file to be loaded.
        
        Returns
        -------
        dataset : An instance of the class GeneralModeDataset 
            Loaded dataset.
        """
        with open(loadpath) as loadfile:
            loaddict = json.load(loadfile)
        
        constructor_dict = {k: v for k, v in loaddict.items() if k in [p.name for p in inspect.signature(cls.__init__).parameters.values()]}
        dataset = cls(**constructor_dict)
        dataset.__dict__['x_array'] = loaddict['z_array']
        return dataset

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
        self.hidden_nodes = np.array(hidden_nodes, dtype = int)
        self.layers = nn.ModuleList()
        for i in range(len(self.hidden_nodes)):
            if i == 0:
                self.layers.append(nn.Linear(1, self.hidden_nodes[i]))
            else:
                self.layers.append(nn.Linear(self.hidden_nodes[i-1], self.hidden_nodes[i]))
        self.layers.append(nn.Linear(self.hidden_nodes[-1], 1))

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()

        # Initialize weights and biases
        #for m in self.modules():
            #if isinstance(m, nn.Linear):
                #nn.init.normal_(m.weight, mean=0, std=1.0e-5)
                #nn.init.constant_(m.bias, val=0)

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

        #F = self.tanh(interm)
        return interm

class Zdot_Encoder(nn.Module):

    def __init__(self, input_channel, channels = [10]):
        super(Zdot_Encoder, self).__init__()

        self.channels = np.array(channels, dtype = int)

        self.layers = nn.ModuleList()
        for i in range(len(self.channels)):
            if i == 0:
                self.layers.append(nn.Conv1d(input_channel, self.channels[i], kernel_size = 3))
            else:
                self.layers.append(nn.Conv1d(self.channels[i-1], self.channels[i], kernel_size = 3))
        self.layers.append(nn.Conv1d(self.channels[i], 1, kernel_size = 1))
        self.dense = nn.Conv1d()
        self.elu = nn.ELU()

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

        # Constant tensors to be used in the model, registered as buffers
        self.register_buffer('C1', torch.tensor([[-1./self.Q, -1.], [1., 0.]]).unsqueeze(0)) # Has size = (1, 2, 2)
        self.register_buffer('C2', torch.tensor([[1.],[0.]]).unsqueeze(0)) # Has size = (1, 2, 1)

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
    def __init__(self, hparams, verbose = True):
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
        self.train_dataset = GeneralModeDataset.load(self.hparams.train_dataset_path)
        ode_params = self.train_dataset.ode_params
        self.ODE = AFM_NeuralODE(**ode_params, hidden_nodes = self.hparams.hidden_nodes)
        #self.noise = nn.Parameter(torch.zeros( (len(self.train_dataset), 1000) ) )
        self.batch_size = self.hparams.batch_size
        self.lr = self.hparams.lr
        self.solver = self.hparams.solver
        self._verbose = verbose
        
        # Compute initial guess for the x0 array and register it as an parameter
        z_arr = self.train_dataset.z_array
        y0_arr = (z_arr[:, 1] - z_arr[:, 0])/0.1
        self.y0 = nn.Parameter(torch.Tensor(y0_arr))

    def forward(self, t, x0, d):
        self.ODE.d = d.view(self.batch_size, 1)

        x_pred = odeint(self.ODE, x0, t, method = self.solver, rtol = 1e-6)
        # x_pred has shape = [time, batch_size, 2]. Permute z_pred so that it has shape = [batch_size, time]
        z_pred = x_pred[:,:,1].permute(1,0)

        return z_pred

    def training_step(self, batch, batch_nb):
        t = batch['time'][0].float()
        d = batch['d'].float()
        z_true = batch['z'].float()
        #y0 = (z_true[:, 1] - z_true[:, 0])/0.1
        x0 = torch.cat([self.y0.view(-1, 1), z_true[:,0].view(-1, 1)], dim = -1)
        #N_data = z_true.size(1) # length of the validation data

        z_pred = self.forward(t, x0, d)
        
        loss_function = nn.MSELoss()

        loss = loss_function(z_pred, z_true)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers = 2, shuffle = False)

    def configure_optimizers(self):
        #optim = torch.optim.Adam(self.parameters(), lr = self.lr)
        optim = torch.optim.LBFGS(self.parameters(), lr = self.lr)
        #sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', 0.5, 100, threshold = 0.05, threshold_mode = 'rel')
        #return [optim], [sched]
        return optim

    def predict_z(self, z):
        pass

    def predict_force(self, d):
        d = np.array(d)
        d_tensor = torch.from_numpy(d).cuda(non_blocking = True).float().reshape(-1, 1)
        F_pred = self.ODE.Fc(d_tensor).cpu().detach().numpy()
        return F_pred

    def TrainModel(self, max_epochs = 10000, checkpoint_path = './checkpoints'):
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
        #logger = TensorBoardLogger(save_dir = os.getcwd(), version = self.slurm_job_id, name = 'lightning_logs')
        checkpoint_callback = ModelCheckpoint(filepath = checkpoint_path, save_top_k = 1, verbose = True, monitor = 'loss', mode = 'min')
        trainer = pl.Trainer(gpus = 1, checkpoint_callback = checkpoint_callback, early_stop_callback = False, max_epochs = max_epochs, log_save_interval = 1)
        trainer.fit(self)

    @classmethod
    def LoadModel(cls, checkpoint_path):
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
        return cls.load_from_checkpoint(checkpoint_path)

if __name__ == '__main__':
    if torch.cuda.is_available:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    savepath = '../Data/digital_snr=1000.json'
    hidden_nodes = torch.Tensor([10, 10, 10])
    hparams = Namespace(**{'lr': 0.01, 'batch_size': 20, 'solver': 'rk4', 'fft_loss': True, 'train_dataset_path': savepath, 'hidden_nodes': hidden_nodes})
    model = LightningTrainer(hparams)

    model.TrainModel()