import numpy as np
from scipy.integrate import solve_ivp
import sys, abc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Later on, create ABCs to wrap AFM model and force models.

class ForcedHarmonicOscillator():
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
        self.Fint = force_model.F
        self.T = 2*Q

    def get_ode(self, d):
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
            dxdt = np.matmul(C1, x) + np.matmul(C2, (d+(self.A0/self.Q)*(np.cos(self.Om*t/(1-0.001))+np.cos((1+0.001)*self.Om*t))+F/self.k))
            return dxdt
        return ode
    
    def solve(self, d, t, x0 = None, **kwargs):
        """
        Solves the ode and returns the solution.

        Parameters
        ----------
        d : float [nm]
            Average tip-sample distance.
        t : 1D numpy array
            Time to evaluate the ode solutions. Must be sorted in increasing order.
        x0 : Numpy array with shape (2, 1)
            Initial value for the state vector. If none is given, x0 = [0, d]. 
        kwargs : dict
            Keyword arguments for scipy.integrate.solve_ivp.
        """
        if x0 == None:
            x0 = np.array([0., d])
            #x0 = np.array([self.Om*self.A0/np.sqrt(self.Q**2*(1-self.Om**2)**2 + self.Om**2), d])
        sol = solve_ivp(self.get_ode(d), (t[0], t[-1]), x0, t_eval = t, vectorized = True, **kwargs)

        return sol

    # Create function for plotting normalized tip-sample force

class TipSampleInteraction(abc.ABC):

    @abc.abstractmethod
    def F(self, x):
        return None

    def PlotForce(self, z_range, zdot_range, n_steps = 1000, figsize = (7, 5), fontsize = 14, **kwargs):
        """
        Plots the tip-sample interaction force as a function of either z, dz/dt, or both.
        """
        assert len(z_range) == 2 and len(zdot_range) == 2, 'z_range and zdot_range must be of the form (start, stop)'
        
        z = np.linspace(*z_range, n_steps)
        zdot = np.linspace(*zdot_range, n_steps)

        x = np.vstack([zdot, z])
        f = self.F(x).flatten()
      
        if z_range[0] == z_range[1]:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            ax.plot(zdot, f, **kwargs)
            ax.set_xlabel('Scaled tip velocity $\omega_0\dot{z} (nm/s)$', fontsize = fontsize)
            ax.set_ylabel('Tip-sample interaction force $F_{int}$(nN)', fontsize = fontsize)
        elif zdot_range[0] == zdot_range[1]:
            fig, ax = plt.subplots(1, 1, figsize = figsize)
            ax.plot(z, f, **kwargs)
            ax.set_xlabel('Tip displacement z (nm)', fontsize = fontsize)
            ax.set_ylabel('Tip-sample interaction force $F_{int}$(nN)', fontsize = fontsize)
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
            ax.scatter(z, zdot, f, **kwargs)
            ax.set_xlabel('Tip displacement z(nm)', fontsize = fontsize)
            ax.set_ylabel('Scaled tip velocity $\omega_0\dot{z} (nm/s)$', fontsize = fontsize)
            ax.set_zlabel('Tip-sample interaction force $F_{int}$(nN)', fontsize = fontsize)

        ax.grid(ls = '--')

        return fig, ax

class Null(TipSampleInteraction):

    def __init__(self):
        pass

    def F(self, x):
        return np.zeros((1, x.shape[-1]))

class DMT_Maugis(TipSampleInteraction):
    """
    Models the tip-sample interaction according to Maugis' approximation to the Derjaguin-Muller-Toporov (a.k.a. Hertz-plus-offset model).
    
    ...

    Attributes
    ----------
    H : float [1e-18 J]
        Hamaker constant of the tip-sample Van-der-Waals interaction.
    R : float [nm]
        Radius of the tip, which is assumed to be spherical.
    z0 : float [nm]
        Distance at which contact is established.
    E : float [GPa]
        Effective Young's modulus between the tip and the sample.
    """

    def __init__(self, H, R, z0, Et, Es, vt, vs):
        """
        Parameters
        ----------
        H : float [1e-18 J]
            Hamaker constant of the tip-sample Van-der-Waals interaction.
        R : float [nm]
            Radius of the tip, which is assumed to be spherical.
        z0 : float [nm]
            Distance at which contact is established.
        Et : float [GPa]
            Young's modulus of the tip.
        Es : float [GPa]
            Young's modulus of the sample.
        vt : float [dimensionless]
            Poisson ratio of the tip.
        vs : float [dimensionless]
            Poisson ratio of the sample.
        """
        self.H = H
        self.R = R
        self.z0 = z0
        self.E = 1/((1-vt**2)/Et + (1-vs**2)/Es)
        
    def F(self, x):
        """
        Computes the force corresponding to the given force model.

        Parameters
        ----------
        x : Numpy array with shape (2, k)
            State vector, where each column corresponds to the form x = [y, z]', where y = dz/dt. 
            k is the number of different x vectors in a single batch.

        Returns
        -------
        F : Numpy array with shape (1, k)
            Force corresponding to state vectors in each columns of the input x.
        """
        F = np.zeros((1, x.shape[-1]))
        # Column indices of state vectors that fulfill the condition z<z0
        contact = x[1, :]<self.z0
        F[0, ~contact] = -self.H*self.R/(6*x[1, ~contact]**2)
        F[0, contact] = (4/3)*self.E*np.sqrt(self.R)*(self.z0 - x[1, contact])**1.5 - self.H*self.R/(6*self.z0**2)

        return F

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
