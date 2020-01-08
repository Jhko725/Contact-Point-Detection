
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ApproachCurve():

    def __init__(self, type_, filepath, z, A, P, f, f0, Q, A0):
        """
        A class to store the approach curve data in. 
        The amplitude (A) and phase (P) are by default mechanical amplitude and phase in units of [m] and [rad].
        
        ...

        Attributes
        ----------
        type_ : "app", "ret", or "both"
            Type of the approach curve. "app" stands for approach curve, "ret" for retraction curve and "both" for both.
            In the case of "both", it is assumed that the approach curve is followed by the retraction curve.         
        filepath : path
            Path where the original data file is stored at. For proxy data with no such file, assign None.
        z : numpy 1D array
            z data of the approach curve in [m].
        A : numpy 1D array
            Mechanical amplitude data of the approach curve in [m].
        P : numpy 1D array
            Mechanical phase data of the approach curve in [rad].
        f : float
            Driving frequency in [Hz].
        f0 : float
            Resonance frequency in [Hz].
        Q : float
            Q-factor of the resonance curve.
        A0 : float
            Mechanical free amplitude in [m].

        Methods
        -------
        __len__()
            Returns the length of the approach curve data.
        __repr__()
            Representation for the ApproachCurve object.
        app()
            Returns the approach part of a given approach curve. Raises AssertionError if called on a retraction-only curve.
        ret()
            Returns the retraction part of a given approach curve. Raises AssertionError if called on a approach-only curve.
        PlotData(figsize, fontsize)
            Plots the amplitude and phase of the given approach using the matplotlib library.
        """
        assert type_ in {'app', 'ret', 'both'}, '''type_ parameter must be 'app', 'ret', or 'both'.'''
        self.type_ = type_
        
        self.filepath = filepath

        # Force z, A, P to be contiguous 1d arrays and f, f0, Q, A0 to be float.
        self.z = np.ravel(z)
        self.A = np.ravel(A)
        self.P = np.ravel(P)
        self.f = float(f)
        self.f0 = float(f0)
        self.Q = float(Q)
        self.A0 = float(A0)

    def __len__(self):
        """
        Returns the length of the approach curve data.
        """
        return len(self.z)

    def __repr__(self):
        """
        Representation for the ApproachCurve object. 
        Without __str__ implemented, this function is called upon print(ApproachCurve instance)
        """
        type_description = {'both': 'full approach curve', 'app': 'approach-only', 'ret': 'retraction-only'}
        return 'An {} data with parameters f = {:8.2f}Hz, f0 = {:8.2f}Hz, Q = {:8.2f}, and A0 = {:8.2e}nm \nRaw data file is at {}'.format(type_description[self.type_], self.f, self.f0, self.Q, self.A0*1e9, self.filepath)

    def app(self):
        """
        Returns the approach part of a given approach curve. Raises AssertionError if called on a retraction-only curve.

        Returns
        -------
        app_component : ApproachCurve instance
            The approach part of the given approach curve.
        """
        assert self.type_ in {'app', 'both'}, '''There is no approach component in a retraction-only curve.'''
        
        if self.type_ == 'app':
            return self
               
        gnd = np.argmin(self.z) # Index corresponding to the smallest z value
        z_app = self.z[:gnd+1]
        A_app = self.A[:gnd+1]
        P_app = self.P[:gnd+1]

        app_component = ApproachCurve("app", self.filepath, z_app, A_app, P_app, self.f, self.f0, self.Q, self.A0)
        return app_component

    def ret(self):
        """
        Returns the retraction part of a given approach curve. Raises AssertionError if called on a approach-only curve.

        Returns
        -------
        ret_component : ApproachCurve instance
            The retraction part of the given approach curve.
        """
        assert self.type_ in {'ret', 'both'}, '''There is no retraction component in an approach-only curve.'''
        
        if self.type_ == 'ret':
            return self
        
        gnd = np.argmin(self.z) # Index corresponding to the smallest z value
        z_ret = self.z[gnd:]
        A_ret = self.A[gnd:]
        P_ret = self.P[gnd:]

        ret_component = ApproachCurve("ret", self.filepath, z_ret, A_ret, P_ret, self.f, self.f0, self.Q, self.A0)
        return ret_component

    def PlotData(self, figsize = (7, 5), fontsize = 14):
        """
        Plots the amplitude and phase of the given approach in the given matplotlib ax object.

        Parameters
        ----------
        figsize : tuple of the form (row, col)
            Size of the resulting matplotlib figure.
        fontsize : float
            Size of the font for the text in the plot.

        Returns
        -------
        fig : matplotlib figure object
            Matplotlib figure corresponding to the plot.
        axes : list of matplotlib axis object
            List containing the matplotlib axes used in the plot. 
            axes[0] corresponds to the z-A plot, and axes[1] corresponds to the z-P plot.
        """
        fig, ax = plt.subplots(1, 1, figsize = figsize)
        ax.plot(self.z, self.A, '.k')
        axx = ax.twinx()
        axx.plot(self.z, self.P, '.r')
        ax.grid(ls = '--')

        ax.set_xlabel('z (nm)', fontsize = fontsize)
        ax.set_ylabel('Amplitude (nm)', fontsize = fontsize)
        axx.set_ylabel('Phase (rad)', fontsize = fontsize)
        
        axes = [ax, axx]

        return fig, axes
    