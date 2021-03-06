import numpy as np
from scipy.integrate import solve_ivp
import sys, abc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .InteractionForce import TipSampleInteraction

class EquationOfMotion(abc.ABC):
    
    @abc.abstractmethod
    def _get_eom(self, d):
        return lambda t, x: None

    @abc.abstractmethod
    def _get_default_x0(self, d):
        """
        Returns the default initial value for the ode problem.

        Returns
        -------
        x0 : Numpy array with shape (m, 1)
            Default initial value for the state vector. m corresponds to the state dimensionality.
        """
        return None

    @abc.abstractmethod
    def _get_default_drive(self):
        return lambda t: None

    @abc.abstractproperty
    def tau(self):
        """
        A read-only property corresponding to the time constant of the given eom model.
        This is used in cases where steady state dynamics are of the interest
        """
        return None
    
    def solve(self, d, t, x0 = None, **kwargs):
        """
        Solves the ode and returns the solution.

        Parameters
        ----------
        d : float [nm]
            Average tip-sample distance.
        t : 1D numpy array
            Time to evaluate the ode solutions. Must be sorted in increasing order.
        x0 : Numpy array with shape (m, 1)
            Default initial value for the state vector. m corresponds to the state dimensionality.
            If none is given, x0 falls back to the result of self._get_x0(). 
        kwargs : dict
            Keyword arguments for scipy.integrate.solve_ivp.
        """

        # If no explicit initial conditions are given, fall back to default initial conditions
        if x0 == None:
            x0 = self._get_default_x0(d)

        sol = solve_ivp(self._get_eom(d), (t[0], t[-1]), x0, t_eval = t, vectorized = True, **kwargs)

        return sol

class ForcedHarmonicOscillator(EquationOfMotion):
    """
    A class to model the AFM QTF/cantilever - sample system as a forced harmonic oscillator subject to a sinusodial driving force and a given tip-sample force F_int.
    Note that in this formulation, rescaled time t_rescaled = omega_0*t is used, and the quantity of interest is the instantaneous tip-sample distance z(t).
    The exact functional form of the tip-sample force must be given during initialization. 
    All units used are rescaled so that 1nm = 1

    ...

    Attributes
    ----------
    Q : float [dimensionless]
        Q-factor of the cantilever/QTF.
    k : float [N/m]
        Force constant of the cantilever/QTF
    Om : float [dimensionless]
        Relative driving frequency of the oscillator - Om = f/f0, where f is the driving freqency and f0 is the resonance frequency
    A0 : float [nm]
        Oscillator amplitude at resonance frequency and without tip-sample force F_int applied to the system.
    F_int : function
        Tip-sample interaction force. Must accept z and dz/dt as input and return a single float as return value.
        The returned force has dimension of [1e-9N].
    T : float [dimensionless]
        Rescaled relaxation time of the cantilever/QTF. 
        T = 2Q, where 2Q/omega_0 is the true relaxation time.
    """

    def __init__(self, Q, k, Om, A0, force_model):
        """
        Parameters
        ----------
        Q : float [dimensionless]
            Q-factor of the cantilever/QTF.
        k : float [N/m]
            Force constant of the cantilever/QTF
        Om : float [dimensionless]
            Relative driving frequency of the oscillator - Om = f/f0, where f is the driving freqency and f0 is the resonance frequency
        A0 : float [nm]
            Oscillator amplitude at resonance frequency and without tip-sample force F_int applied to the system.
        F_int : function
            Tip-sample interaction force. Must accept z and dz/dt as input and return the corresponding tip-sample force.
            The returned force has dimension of [1e-9N].
        """
        self.Q = Q
        self.k = k
        self.Om = Om
        self.A0 = A0
        assert issubclass(type(force_model), TipSampleInteraction), "F_int must be a TipSampleInteraction!"
        self.Fint = force_model

    def _get_eom(self, d, F_drive = None):
        """
        Returns the corresponding ode function of the model. 
        x is a state vector, where each column corresponds to the form x = [y, z]', where y = dz/dt. 
        t is the rescaled time of the form t_rescaled = t_true * omega_0.

        Parameters
        ----------
        t : float [dimensionless]
            Rescaled time, given by t_rescaled = t_true * omega_0, where omega_0 is the angular resonance frequency.
        x : Numpy array with shape (2, k)
            State vector, where each column corresponds to the form x = [y, z]', where y = dz/dt. 
            k is the number of different x vectors in a single batch.
        d : float [nm]
            Average tip-sample distance.

        Returns
        -------
        dxdt : Numpy array with shape (2, k)
            State vector, where each column corresponds to the form dxdt = [dydt, dzdt]'
        """
        # Assignments for better readability
        Q = self.Q
        k = self.k

        # Coefficient matrices
        C1 = np.array([[-1/Q, -1], [1, 0]], dtype = float)
        C2 = np.array([1, 0], dtype = float).reshape(2, 1)
        
        # Check forces, assign default to F_drive if passed None
        
        if F_drive == None:
            F_drive = self._get_default_drive()

        def eom(t, x):
            Fd = F_drive(t)
            # Force Fts to be two-dimensional, with the second dimension being the batch size
            Fts = self.Fint(x).reshape(1, -1)

            dxdt = np.matmul(C1, x) + np.matmul(C2, (d + Fd/k + Fts/k))
            return dxdt
        return eom
    
    def _get_default_x0(self, d):
        return np.array([0., d])

    def _get_default_drive(self):
        return lambda t: (self.k*self.A0/self.Q)*np.cos(t)

    @property
    def tau(self):
        return 2*self.Q

    # Create function for plotting normalized tip-sample force

class BimodalFHO(EquationOfMotion):

    def __init__(self, Q0, Q1, k1, k2, Om, A00, A01, force_model):
        self.Q0 = Q0
        self.Q1 = Q1
        self.k1 = k1
        self.k2 = k2
        self.Om = Om
        self.A00 = A00
        self.A01 = A01
        self.Fint = force_model
    
    def _get_eom(self, d):
        # Look into the equation. Is Q1 correct?
        C1 = np.array([[-1/self.Q1, -1, 0, 0], [1, 0, 0, 0], [0, 0, -self.Om/self.Q1, -self.Om**2], [0, 0, 1, 0]], dtype = float)
        C2 = np.array([[1], [0], [0], [0]], dtype = float)
        C3 = np.array([[0], [0], [1], [0]], dtype = float)

        def eom(t, x):
            d_state = np.zeros(x[0:2, :].shape)
            d_state[1, :] = d_state[1, :] + d
            z_state = x[0:2, :] + x[2:, :] + d_state
            F = self.Fint(z_state)

            dxdt = np.matmul(C1, x) + np.matmul(C2, ((self.A00/self.Q0)*np.cos(t) + (self.A01/self.Q1)*np.cos(self.Om*t) + F/self.k1)) + np.matmul(C3, ((self.A00/self.Q0)*np.cos(t) + (self.A01/self.Q1)*np.cos(self.Om*t) + F/self.k2))
            return dxdt
        return eom

    def _get_default_x0(self, d = None):
        return np.array([0., self.A00, 0., self.A01])

    @property
    def tau(self):
        return 2*np.max(self.Q0, self.Q1)

def SimulateGeneralMode(AFM, d_array, dt, N_data, relaxation = 7, x0 = None, **kwargs):
    """
    Creates the general mode AFM approach curve according to the given AFM model.
    For each average tip-sample distance d in d_array, the steady state trajectory of the tip is calculated.

    Parameters
    ----------
    AFM : an instance of a class modeling the AFM
        The AFM model to be used in simulating the tip dynamics.
    d_array : 1D numpy array
        An array of average tip-sample distances for the approach curve.
    dt : float
        Time increment for the cantilever trajectory z(t).
    N_data : int
        Number of steady state trajectory data to be generated per average tip-sample distance d.
    relaxation : int
        How many multiples of the time constant to be discarded prior to sampling the steady state dynamics.
    kwargs : dict
        Keyword arguments for scipy.integrate.solve_ivp.
    
    Returns
    -------
    t : numpy 1D array
        Time array used to solve the ode
    x_array : numpy 3D array with dimensions (len(d_array), 2, N_data)
        Simulated general mode approach curve data. 
        The first dimension corrresponds to a given average tip-sample distance d.
        The second dimension corresponds to the dimension of the state vector x in the form (z_dot, z)
        The last dimension is the time series data dimension
    """
    # Number of data points needed for relaxation
    N_relax = np.ceil(AFM.tau*relaxation/dt)
    t = np.arange(N_relax+N_data)*dt

    d_array = np.array(d_array)
    x_array = np.zeros((d_array.size, 2, N_data))

    sys.stdout.write('Data generation started\n')
    for i in range(d_array.size):
        sol = AFM.solve(d_array[i], t, x0 = x0, **kwargs)
        x_array[i, :, :] = sol.y[:, -N_data:]
    
        sys.stdout.write('\r')
        sys.stdout.write('{:d}/{:d} generated'.format(i+1, d_array.size))
        sys.stdout.flush()

    return t, x_array
