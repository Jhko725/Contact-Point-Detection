import numpy as np
import matplotlib.pyplot as plt
import abc
from numba import vectorize, float64
import functools

class TipSampleInteraction(abc.ABC):

    def __init__(self):
        self._F = self._get_F()

    def __neg__(self):
        return NegateForce(self)

    def __sum__(self, other):
        return SumForce([self, other])

    @abc.abstractmethod
    def _get_F(self):
        return lambda x, y: None

    def __call__(self, x):
        return self._F(x[1,:], x[0,:])

    def PlotForce(self, z_range, zdot_range, n_steps = 1000, figsize = (7, 5), fontsize = 14, **kwargs):
        """
        Plots the tip-sample interaction force as a function of either z, dz/dt, or both.
        """
        assert len(z_range) == 2 and len(zdot_range) == 2, 'z_range and zdot_range must be of the form (start, stop)'
        
        z = np.linspace(*z_range, n_steps)
        zdot = np.linspace(*zdot_range, n_steps)

        x = np.vstack([zdot, z])
        f = self(x).flatten()
      
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

class NullForce(TipSampleInteraction):

    def __init__(self):
       self._F = self._get_F()

    def _get_F(self):
        @vectorize([float64(float64, float64)])
        def _F(z = None, zdot = None):
            return 0
        return _F

class ConstantForce(TipSampleInteraction):

    def __init__(self, F0):
        self.F0 = F0
        self._F = self._get_F()

    def _get_F(self):
        @vectorize([float64(float64, float64)])
        def _F(z = None, zdot = None):
            return self.F0
        return _F

class NegateForce(TipSampleInteraction):

    def __init__(self, force):
        assert issubclass(type(force), TipSampleInteraction), "Input force must be a TipSampleInteraction!"
        self.original_force = force
        self._F = self._get_F()

    def _get_F(self):
        return lambda z, zdot: -self.original_force._F(z, zdot)

class SumForce(TipSampleInteraction):

    def __init__(self, force_list):
        for force in force_list:
            assert issubclass(type(force), TipSampleInteraction), "Input force must be a TipSampleInteraction!"
        self.force_list = force_list
        self._F = self._get_F()

    def _get_F(self):
        def _F(z, zdot):
            F_list = [force._F(z, zdot) for force in self.force_list]
            return sum(F_list)
        return _F

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
        self._F = self._get_F()

    def _get_F(self):
        z0 = self.z0
        H = self.H
        R = self.R
        E = self.E

        @vectorize([float64(float64, float64)])
        def _F(z, zdot = None):
            if z > z0:
                return -H*R/(6*z**2)
            else:
                return (4/3)*E*np.sqrt(R)*(z0 - z)**1.5 - H*R/(6*z0**2)
        return _F

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

    def _z_off(self):
        gamma_sv = self.H/(24*np.pi*self.z0**2)
        r = (3*np.pi*gamma_sv*self.R**2/self.E)**(1/3)
        V = 4*np.pi*self.R*self.h + (4/3)*np.pi*self.h**3 + 2*np.pi*r**2*self.h
        z_off = V**(1/3) - V**(2/3)/(5*self.R)
        return z_off

    def _get_F(self):

        R = self.R
        h = self.h
        gamma_lv = self.gamma_lv
        app = self.app

        z_on = 2*self.h
        z_off = self._z_off()
        
        @vectorize([float64(float64, float64)])
        def _F(z, zdot = None):
            if app:
                return -4*np.pi*gamma_lv*R/(1 + z/h) if z<z_on else 0
            else:
                return -4*np.pi*gamma_lv*R/(1 + z/h) if z<z_off else 0
        return _F    