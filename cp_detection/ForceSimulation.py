import numpy as np
from scipy.integrate import solve_ivp

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
    """

    def __init__(self, Q, k, Om, A0, Fint):
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
        C1 : Numpy array with shape (2, 2)
            Coefficient matrix for the ode of the form C1 = [[-1./self.Q, -1.], [1., 0.]]
        C2 : Numpy array with shape (2, 1)
            Coefficient matrix for the ode of the form C2 = [[1.], [0.]]
        """
        self.Q = Q
        self.k = k
        self.Om = Om
        self.A0 = A0
        self.Fint = Fint

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
            dxdt = np.matmul(C1, x) + np.matmul(C2, (d+(self.A0/self.Q)*np.cos(self.Om*t)+F/self.k))
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
            Initial value for the state vector. If none is given, x0 = [Om*A0/sqrt(Q^2(1-Om^2)^2 + Om^2), d]. 
        kwargs : dictionary
            Keyword arguments for scipy.integrate.solve_ivp.
        """
        if x0 == None:
            x0 = np.array([self.Om*self.A0/np.sqrt(self.Q**2*(1-self.Om**2)**2 + self.Om**2), d])
        sol = solve_ivp(self.get_ode(d), (t[0], t[-1]), x0, t_eval = t, vectorized = True, **kwargs)

        return sol

class DMT_Maugis():
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