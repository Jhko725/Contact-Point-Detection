import glob
import os
import json
import numpy as np
from .ApproachCurve import ApproachCurve

def FileParser(data_dir, data_type, data_format = 'txt'):
    """
    Returns the list of paths of all measurement data in the subdirectories of data_dir corresponding to the given data_type and data_format.

    Parameters
    ----------
    data_dir: path
        Directory for the measurement data. Note that all data must be stored in the direct subdirectory of data_dir.
    data_type: "app" or "res"
        Corresponds to the type of the measurement. "app" for approach curves and "res" for resonance curves.
    data_format: "str"
        The file extension of the desired data. "txt" by default. 

    Returns
    -------
    files: list
        A list of filepaths corresponding to the desired data_type and data_format
    """

    type_string = {'res': '*Resonance*.', 'app': '*Approach*.'}

    path = os.path.join(data_dir, '*', type_string[data_type] + data_format)

    files = glob.glob(path, recursive = True)

    print('{} files loaded'.format(len(files)))
    
    return files

class Json2App():
    """
    A class to convert json format approach curve raw data into instances of the class ApproachData. 
    
    ...

    Attributes
    ----------
    filepath : path
        Path to the json file to be converted.
    z : numpy 1D array
        Measured z data in units of [m].
    Ar : numpy 1D array
        Measured raw amplitude data in units of [V_lockin]. 
        [V_lockin] is the voltage recorded by the lock_in amplifier.
    Pr : numpy 1D array
        Measured raw phase data in units of [V_lockin]. 
        [V_lockin] is the voltage recorded by the lock_in amplifier.
    
    
    Methods
    -------
    _CalibPhasOffset()
        Calculates the phase offset according to the formula of M. Lee et al, App. Phys. Lett. 91, 023117(2007).
    _CalcElecPhase(Pe0)
        Converts the raw phase data Pr[V_lockin] to the electrical phase data Pe[rad]. 
        The conversion formula is: Pe = Pr * pi/10 + Pe0, where Pe0 is the electrical phase offset.
    """    

    def __init__(self, jsonpath, eval_res_params = False, respath = None, eval_phas_offset = False):
        """
        Parameters
        ----------
        jsonpath : path
            Path to the json file to be converted.
        eval_res_params : bool
            Boolean flag to determine whether to compute the resonance curve fitting parameters or not. Default: False.
            If this flag is set to True or the json file does not contain any one of the fields "w0", "Q", "C0_C" and "I0_r", the resonance curve fiting parameters are automatically computed.
        respath : path
            Path to the resonance curve file corresponding to the approach curve data. 
            Only required if eval_res_params == True.
        eval_phas_offset : bool
            Boolean flag to determine whether to compute the electric phase offset Pe0 or not. Default: False.
            If this flag is set to True or the json file does not contain the field "phas_offset", Pe0 is automatically computed.
        """
        self.filepath = jsonpath

        with open(jsonpath) as json_file:
            self.json_data = json.load(json_file)
        
        self.f = self.json_data['w'] # driving frequency
        self.z = np.array(self.json_data['z'])

        # Retrieve resonance curve parameters
        if not all(param in self.json_data.keys() for param in ['w0', 'Q', 'C0_C', 'I0_r']) or eval_res_params:
            self.res_params = self._CalcResParams()
        else:
            self.res_params = {'f0': self.json_data['w0'], 'Q': self.json_data['Q'], 'C0_C' : self.json_data['C0_C'], 'I0_raw': self.json_data['I0_r']}

        # Calculate electric amplitude and phase
        if 'phas_offset' not in self.json_data.keys() or eval_phas_offset:
            Pe0 = self._CalcPhaseOffset()
        else:
            Pe0 = self.json_data['phas_offset']
        self.Pe = self._CalcElecPhase(Pe0)
        self.Ae = self._CalcElecAmp()

        # Calculate mechanical amplitude and phase
        self.Am, self.Pm, self.A0 = self._CalcMechAmpPhas()
        
    def _CalcResParams(self):
        return 0
    
    def _CalcPhaseOffset(self):
        return 0

    def _CalcElecPhase(self, Pe0):
        """
        Converts the raw phase data Pr[V_lockin] to the electrical phase data Pe[rad]. 
        The conversion formula is: Pe = Pr * pi/10 + Pe0, where Pe0 is the electrical phase offset.
        
        Parameters
        ----------
        Pe0 : float
            Electrical phase offset, calculated according to M. Lee et al, App. Phys. Lett. 91, 023117(2007).
        
        Returns
        -------
        Pe : numpy 1D array
            Electrical phase data in [rad].
        """
        Pr = np.array(self.json_data['phas_r'])
        Pe = Pr*np.pi/10 + Pe0

        return Pe

    def _CalcElecAmp(self):
        """
        Converts the raw amplitude data Ar[V_lockin] to the electrical amplitude data Ae[A].
        The conversion is done as the following:
        1) Ar : ( -10[V_lockin], 10[V_lockin] ) -> ( -SEN[V], SEN[V] ), where SEN[V] is the lock in amplifier sensitivity.
        2) ( -SEN[V], SEN[V] ) -> Ae : (-SEN[V] * IVgain[A/V], SEN[V] * IVgain[A/V]), where IVgain[A/V] is the gain of the AFM amplifier circuit.
        
        Returns
        -------
        Ae : numpy 1D array
            Electric amplitude data in [A].
        """
        SEN = self.json_data['sens']
        IVgain = self.json_data['IVgain']

        Ar = np.array(self.json_data['amp_r'])
        Ae = (Ar/10)*SEN/IVgain

        return Ae

    def _CalcMechAmpPhas(self):
        # Convert I0_raw[V] into I0[A]
        SEN = self.json_data['sens']
        IVgain = self.json_data['IVgain']
        I0 = (self.res_params['I0_raw']/10)*SEN/IVgain
        
        # See M. Lee et al, App. Phys. Lett. 91, 023117(2007)
        wC0V0 = (self.res_params['C0_C']*I0/self.res_params['Q'])*self.f/self.res_params['f0']
        
        # first, calculate mechanical amplitude in (A), then convert to (m) using the piezoCf(m/A)
        Am = np.sqrt(self.Ae**2 - 2*wC0V0*self.Ae*np.sin(self.Pe) + wC0V0**2)*self.json_data['piezoCf']
        
        # calculate mechanical phase in (rad)
        Pm = np.arctan2(self.Ae*np.sin(self.Pe)-wC0V0, self.Ae*np.cos(self.Pe))

        # Also calculate the free amplitude
        A0 = I0*self.json_data['piezoCf']

        return Am, Pm, A0

    def __call__(self, type_ = 'both'):
        app_curve = ApproachCurve(type_, self.filepath, self.z, self.Am, self.Pm, self.f, self.res_params['f0'], self.res_params['Q'], self.A0)
        return app_curve
