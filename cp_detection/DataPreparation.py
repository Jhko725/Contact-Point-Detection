from .ApproachCurve import ApproachCurve
from .FileParse import Json2App
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

class AppCurveDataset(Dataset):
    """
    A dataset class that holds the processed approach curve data. Inherits from torch.utils.data.Dataset.
    To change how an ApproachCurve instance is modified into a dataset element, change the _PrepOne function.
    
    ...

    Attributes
    ----------
    appcurve_list : a list of ApproachCurve instances used to create the dataset

    Methods
    -------
    __len__()
        Returns the number of elements in the dataset.
    __getitem(idx)
        Returns the idx-th element of the dataset.
    _PrepOne(appcurve)
        A staticmethod that changes a given ApproachCurve instance into a dataset element.
    PlotParamDist()
        Returns the histograms of the approach curve parameters (f0, f/f0, Q, A0) in the dataset.
    """
    
    def __init__(self, json_list, type_ = "both"):
        # Create a list of z-increasingly sorted approach-only curves
        self.appcurve_list = [Json2App(jsonfile)(type_).app() for jsonfile in json_list]
        for app in self.appcurve_list:
            app.SortData()
        
    def __len__(self):
        return len(self.appcurve_list)

    def __getitem__(self, idx):
        return None
    
    @staticmethod
    def _PrepOne(appcurve):
        pass

    def PlotParamDist(self, figsize = (16, 12), fontsize = 14):
        fig, axes = plt.subplots(2, 2, figsize = figsize)
        
        f_arr = np.array([app.f for app in self.appcurve_list])
        f0_arr = np.array([app.f0 for app in self.appcurve_list])
        Q_arr = np.array([app.Q for app in self.appcurve_list])
        A0_arr = np.array([app.A0*1e9 for app in self.appcurve_list])

        axes[0][0].hist(f0_arr)
        axes[0][0].set_title('Histogram of resonance frequency: $f_0$ [Hz]', fontsize = fontsize)
        axes[0][1].hist(f_arr/f0_arr)
        axes[0][1].set_title('Histogram of relative driving frequency: $f_0/f$', fontsize = fontsize)
        axes[1][0].hist(Q_arr)
        axes[1][0].set_title('Histogram of Q factor $Q$', fontsize = fontsize)
        axes[1][1].hist(A0_arr)
        axes[1][1].set_title('Histogram of free amplitude $A_0$ [nm]', fontsize = fontsize)

        for ax in axes.flatten():
            ax.grid(ls = '--')

        return fig, axes