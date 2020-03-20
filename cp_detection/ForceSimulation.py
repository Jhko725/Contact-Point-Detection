import numpy as np
from scipy.integrate import solve_ivp
import sys, abc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        self.Fint = force_model
        self.T = 2*Q

    def _get_eom(self, d):
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
        C1 = np.array([[-1./self.Q, -1.], [1., 0.]])
        C2 = np.array([[1.], [0.]])
        
        def ode(t, x): 
            F = self.Fint(x)
            dxdt = np.matmul(C1, x) + np.matmul(C2, (d+(self.A0/self.Q)*(np.cos(self.Om*t))+F/self.k))
            return dxdt
        return ode
    
    def _get_default_x0(self, d):
        return np.array([0., d])

    # Create function for plotting normalized tip-sample force

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
    z_array : numpy 2D array with dimensions (len(d_array), N_data)
        Simulated general mode approach curve data. 
        Each row corrresponds to data for a given average tip-sample distance d.
    """
    # Number of data points needed for relaxation
    N_relax = np.ceil(AFM.T*relaxation/dt)
    t = np.arange(N_relax+N_data)*dt

    d_array = np.array(d_array)
    z_array = np.zeros((d_array.size, N_data))

    sys.stdout.write('Data generation started\n')
    for i in range(d_array.size):
        sol = AFM.solve(d_array[i], t, x0 = x0, **kwargs)
        z_array[i, :] = sol.y[1, -N_data:]
    
        sys.stdout.write('\r')
        sys.stdout.write('{:d}/{:d} generated'.format(i+1, d_array.size))
        sys.stdout.flush()

    return t, z_array
