
import numpy as np
import pandas as pd

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
        return len(self.z)

    def __repr__(self):
        type_description = {'both': 'full approach curve', 'app': 'approach-only', 'ret': 'retraction-only'}
        return 'An {} data with parameters f = {%8.2f} Hz, f0 = {%8.2f} Hz, Q = {%8.2f}, and A0 = {%8.2e} nm'.format(type_description[self.type_], self.f, self.f0, self.Q, self.A0*1e9)

    def app(self):
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
        assert self.type_ in {'ret', 'both'}, '''There is no retraction component in an approach-only curve.'''
        
        if self.type_ == 'ret':
            return self
        
        gnd = np.argmin(self.z) # Index corresponding to the smallest z value
        z_ret = self.z[gnd:]
        A_ret = self.A[gnd:]
        P_ret = self.P[gnd:]

        ret_component = ApproachCurve("ret", self.filepath, z_ret, A_ret, P_ret, self.f, self.f0, self.Q, self.A0)
        return ret_component