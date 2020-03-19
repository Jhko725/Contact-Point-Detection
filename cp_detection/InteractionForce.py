import numpy as np
import matplotlib.pyplot as plt
import abc

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
        iscontact = x[1, :]<self.z0
        F[0, ~iscontact] = -self.H*self.R/(6*x[1, ~iscontact]**2)
        F[0, iscontact] = (4/3)*self.E*np.sqrt(self.R)*(self.z0 - x[1, iscontact])**1.5 - self.H*self.R/(6*self.z0**2)

        return F

class Capillary(TipSampleInteraction):
    """
    Models the capillary force due to the formation of a water nano-meniscus between the tip and the sample.
    The derivations are found in L. Zitzler, S. Herminghaus, and F. Mugele, Phys. Rev. B, 66, 155436 (2002).
    """

    def __init__(self, H, R, z0, Et, Es, vt, vs, h, gamma_lv, app):
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
        h : float [nm]
            Thickness of the hydration layer. Note that for the model to hold, h > z0 should be satisfied.
        gamma_lv : float [J/m^2]
            Surface tension (or liquid-vapor surface energy) of the liquid forming the capillary bridge.
        app : bool
            True if the tip is approaching the surface, and False if retracting.
        """
        self.H = H
        self.R = R
        self.z0 = z0
        self.h = h
        self.gamma_lv = gamma_lv
        self.app = app
        self.E = 1/((1-vt**2)/Et + (1-vs**2)/Es)

        self.z_on = 2*h
        self.z_off = self._z_off()

    def _z_off(self):
        gamma_sv = self.H/(24*np.pi*self.z0**2)
        r = (3*np.pi*gamma_sv*self.R**2/self.E)**(1/3)
        V = 4*np.pi*self.R*self.h + (4/3)*np.pi*self.h**3 + 2*np.pi*r**2*self.h
        z_off = V**(1/3) - V**(2/3)/(5*self.R)
        return z_off

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
        iscapill = x[1, :]<self.z_on if self.app else x[1, :]<self.z_off

        F[0, iscapill] = -4*np.pi*self.gamma_lv*self.R/(1 + x[1, iscapill]/self.h)
        
        return F