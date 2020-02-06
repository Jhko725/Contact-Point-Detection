import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline , Rbf, UnivariateSpline
from scipy.integrate import simps
from tqdm import tqdm
import json
import glob, os
import quadpy

### This module imports all data and returns :\
# 0) appended dictionary of keys
#   i)F_L : list
#           Conservative force using formula by Lee et al.(1st order) \
#   ii)F_D : list
#           Conservative force using formula by Dagdeviren et al. \
#   iii)U_D : list
#           Potential energy using formula by Dagdeviren et al.\
#   iv) cp_DMT : list
#           Contact point using DMT contact model, standard deviation error of a fit\
#   v) cp_JKR : list
#           Contact point using JKR contact model, standard deviation error of a fit\
#1)Force-distance curves using two formula w/ & w/o cp demarcation

## Inputs
# Designate data folder for mechanical amplitude/phase curves in json format 
folderpath = "./data/sim/"
vispath = folderpath+"vis/"
datatype = "sim"
xlim = 5
sm = 0
exp_param = {'zc' : 0.3,  'H' : 0.2, 'Eₜ' : 72.5, 'R' : 80.,\
    'νₜ' : 0.17, 'Eₛ' : 200., 'νₛ' : 0.25, 'r_rod' : 3000, 'l_rod' : 151.1*1e3}

## Functions
# Numerical Differentiation
def NumDiff (F, x):
    N = len(x)
    f= np.zeros(N)
    # Internal points
    for i in range(1,N-1):
        f[i] = (F[i+1]*(x[i]-x[i-1])**2 + F[i]*(x[i+1]-x[i-1])*(x[i+1]+x[i-1]-2*x[i])-F[i-1]*(x[i+1]-x[i])**2)/((x[i+1]-x[i-1])*(x[i+1]-x[i])*(x[i]-x[i-1]))
        #f[i] = (R = R, H = H, Eₜ = Eₜ, νₜ = νₜ, Eₛ = Eₛ, νₛ = νₛF[i+1]-F[i-1])/(x[2]-x[0])
    # End points
    f[0] = (F[1]-F[0])/(x[1]-x[0])
    f[N-1] = (F[N-1]-F[N-2])/(x[N-1]-x[N-2])
    return f

def DMT(z, zc = 0.3, R = 80., H = 0.2, Eₜ = 72.5, νₜ = 0.17, Eₛ = 200., νₛ = 0.25, η =0.): 
    z= np.asarray(z)
    z_nc= z[z>zc]
    z_c = z[z<=zc]
    E = 1/((1-νₜ**2)/Eₜ + (1-νₛ**2)/Eₛ) # Effective modulus of tip and sample

    return np.append(-H*R/(6*zc**2) + (4/3)*E*np.sqrt(R)*(zc-z_c)**1.5,-H*R/(6*z_nc**2) )

def DMT_r(z, z_off, fmin, R, zc = 0.3, H = 0.2, Eₜ = 72.5, νₜ = 0.17, Eₛ = 200., νₛ = 0.25): 
    E = 1/((1-νₜ**2)/Eₜ + (1-νₛ**2)/Eₛ) # Effective modulus of tip and sample
    return -H*R/(6*zc**2) + (4/3)*E*np.sqrt(R)*(zc-(z+z_off))**1.5 + fmin

def DMTcap_r(z, z_off, fmin, R, zc = 0.3, H = 0.2, Eₜ = 72.5, νₜ = 0.17, Eₛ = 200., νₛ = 0.25): 
    E = 1/((1-νₜ**2)/Eₜ + (1-νₛ**2)/Eₛ) # Effective modulus of tip and sample
    return -H*R/(6*zc**2) + (4/3)*E*np.sqrt(R)*(zc-(z+z_off))**1.5-4*np.pi*0.072*R/(1+(z+z_off)/0.74)+ fmin

def DMTcaprod_r(f, z_off, fmin, R, zc = 0.3, H = 0.2, Eₜ = 72.5, νₜ = 0.17, Eₛ = 200., νₛ = 0.25, r_rod = 4000, l_rod = 151.1*1e3): 
    E = 1/((1-νₜ**2)/Eₜ + (1-νₛ**2)/Eₛ) # Effective modulus of tip and sample
    return -(3*(f-fmin)/(4*np.pi*E*R**0.5))**(2/3) - (f-fmin)/(np.pi*r_rod**2)*l_rod/Eₜ + zc - z_off

'''    
def JKR(f, c1, c2, c3, c4,c5):
    return -(c2/c3*(c4**0.5+(f+c4+c5)**0.5)**2)/c2+4/3*((c2/c3*(c4**0.5+(f+c4+c5)**0.5)**2)**0.5*c4/c2/c3)**0.5+c1 #JKR type force with c(1)=contact, c(2)= radius of tip c(3)=elastic modulus c(4)=maximum adhesive force

def DMT_cap(z, c1, c2,c3, c4):
    return -H*c2/6/(H/24/np.pi/c3)+4/3*E*c2**0.5*(c1+(H/24/np.pi/c3)**0.5-z)**1.5-4*np.pi*c3*c2/(1+(z-c1)/h)+c4  #DMT type force with c(1)=contact, c(2)= radius of tip c(3)=effective surface energy
'''

# Import all files in the folder for analysis
filename_ls = []
main_dir = os.getcwd()
os.chdir(folderpath)
print('Initiallizing analysis using following data :')
for index, file in enumerate(glob.glob("*.json")):
    filename_ls.append(file)
    print(f'{index:4d} ===> {file:15}')
os.chdir(main_dir)


for filename in tqdm(filename_ls):
    with open(folderpath + filename, "r") as data_json:
        data = json.load(data_json)
    z= data["z"]
    amp = data["amp"]
    phas = data["phas"]
    Q = data["Q"]
    Ω = data["Omega"]
    k = data["k"]
    A0= data["A0"]

    ## Preprocess data
    # Reorder of the sequence
    if z[0]<z[1]:
        data["z"] = z = z[::-1]
        data["amp"] = amp = amp[::-1]
        data["phas"] = phas = phas[::-1]
    # Amplitude - distance conversion for experimental data
    if datatype == 'exp':
        amp = [i*1e9 for i in amp]
        z = [i*1e9 for i in z]
        A0 = A0*1e9
    # Phase conversion for simulation data
    if datatype == 'sim' :
        phas = [np.pi/2-i for i in phas]
    # Add variables
    if 'gnd' in data:
        gnd = data["gnd"]
    else :
        data["gnd"] = gnd = z.index(min(z))
    
    if 'k_int' in data:
        k_int = data["k_int"] 
    else :
        data["k_int"] = k_int = [k*(Ω/Q*(A0/amp[i])*np.sin(phas[i])+\
            (1-Ω**2)*(A0/amp[i]*np.cos(phas[i])-1)) for i in range(len(z))]
    
    ## Process data
    # Trim and reorder data sequence from nearest -> farthest
    z = z[:gnd][::-1]
    amp = amp[:gnd][::-1]
    phas = phas[:gnd][::-1]
    k_int = k_int[:gnd][::-1]
    N = len(z)

    #Plot Amplitude Phase Curve
    amp_rbf = Rbf(z, amp, smooth = sm)
    phas_rbf = Rbf(z, phas, smooth = sm)
    fig, axes = plt.subplots(figsize=(8,6))
    l_a = axes.plot(z,[i/A0 for i in amp], label = "Amplitude \n A0 = {0:.4f} nm".format(A0), color = 'k', ls = '-', marker= '.')
    l_arbf = axes.plot(z,[i/A0 for i in amp_rbf(z)], label = "Rbf smooth = "+str(sm), color = 'k', ls = '--', marker= '.')
    plt.ylabel('Amplitude(rel.)')
    axes_ = axes.twinx()
    l_p = axes_.plot(z,[i for i in phas], label= "Phase", color = 'b', ls='-', marker = '.')
    l_prbf = axes_.plot(z,[i for i in phas_rbf(z)], label= "Rbf smooth = "+str(sm), color = 'b', ls='--', marker = '.')
    lns = l_a+l_p +l_arbf + l_prbf
    labs = [l.get_label() for l in lns]
    plt.ylabel('Phase(rad)')
    plt.xlabel('Distance(nm)')
    plt.xlim([0,xlim])
    plt.legend(lns, labs, loc = 'lower right')
    plt.grid(ls='--')
    plt.savefig(vispath+filename.split('.')[0]+'_AmpPhas.png') 
    plt.show()


    k_int = [k*(Ω/Q*(A0/amp_rbf(z[i]))*np.sin(phas_rbf(z[i]))+\
            (1-Ω**2)*(A0/amp_rbf(z[i])*np.cos(phas_rbf(z[i]))-1)) for i in range(len(z))]
    ##Force reconstruction using formula by Lee et al.
    F_k = np.zeros(N)
    for i in range(N-1):
        F_k[i] = simps(k_int[i:N-1], z[i:N-1])
    F_k[N-1] = F_k[N-2]
    cp = np.argmin(F_k)-int(0/(z[1]-z[0]))
    cp0 = np.argmin(F_k)
    #fit_param_DMT, var_DMT = curve_fit(lambda z, z_off, fmin, R  : DMT_r (z, z_off, fmin, R, zc = exp_param['zc'], H = exp_param['H'],\
    #     Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), z[:cp], F_k[:cp], p0 =[exp_param['zc']-z[cp], \
    #         -DMT(exp_param['zc'])+F_k[cp], exp_param['R']] )
    fit_param_DMT, var_DMT = curve_fit(lambda z, z_off, fmin  : DMT_r (z, z_off, fmin, zc = exp_param['zc'], R = exp_param['R'], H = exp_param['H'],\
         Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), z[:cp], F_k[:cp], p0 =[exp_param['zc']-z[cp], \
             -DMT(exp_param['zc'])+F_k[cp]] )
    #fit_param_DMT, var_DMT = curve_fit(lambda f, z_off, fmin  : DMTcaprod_r (f, z_off, fmin, zc = exp_param['zc'], R = exp_param['R'], H = exp_param['H'],\
    #    Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ'],r_rod= exp_param['r_rod'], l_rod= exp_param['l_rod']),\
    #    F_k[:cp], z[:cp],p0 =[exp_param['zc']-z[cp], F_k[cp]] )



    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(z,F_k, label = "Conservative Force(Lee $1^{st}$)", color = 'k')
    #axes.plot(z[:cp],DMT_r(z[:cp],z_off = fit_param_DMT[0], fmin = fit_param_DMT[1],R = fit_param_DMT[2],  zc = exp_param['zc'], H = exp_param['H'],\
    #     Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), label = "DMT, r = "+str(fit_param_DMT[2]), color = 'r')
    axes.plot(z[:cp0],DMT_r(z[:cp0],z_off = fit_param_DMT[0], fmin = fit_param_DMT[1], zc = exp_param['zc'], R= exp_param['R'],H = exp_param['H'],\
         Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), label = "DMT fit", color = 'r')
    axes.axvline(x = exp_param['zc']- fit_param_DMT[0], linestyle = '-', color = 'r',linewidth = 5, label = 'contact point(fit) : {0:.4f}'.format(exp_param['zc']- fit_param_DMT[0]))
    axes.axvspan(exp_param['zc']-fit_param_DMT[0]-np.sqrt(np.diag(var_DMT))[0],exp_param['zc']- fit_param_DMT[0]+np.sqrt(np.diag(var_DMT))[0] , facecolor='r', alpha=0.5)
    #axes.plot(DMTcaprod_r(F_k[:cp],z_off = fit_param_DMT[0], fmin = fit_param_DMT[1], zc = exp_param['zc'], R= exp_param['R'],H = exp_param['H'],\
    #     Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ'],r_rod= exp_param['r_rod'], l_rod= exp_param['l_rod'])\
    #         ,z[:cp], label = "DMT, r = "+str(fit_param_DMT), color = 'r')
    if 'force_params' in data:
        force_params = data["force_params"]
        axes.plot(z,DMT(z,zc = force_params['zc'], R = force_params['R'], H = force_params['H'],\
            Eₜ = force_params['Eₜ'], νₜ = force_params['νₜ'], Eₛ = force_params['Eₛ'], \
            νₛ = force_params['νₛ'], η = force_params['η']), color = 'r', ls = '--', label = 'Model Force')
        axes.axvline(x =exp_param['zc'], linestyle = '--', color = 'r',linewidth = 5, label = 'contact point(model)')
    plt.xlim([0,xlim])
    #plt.ylim([F_k[np.argmin(F_k)]-5,max(F_k[int(xlim/(z[1]-z[0]))]+5,F_k[0]+5)])
    plt.ylim([-40,100])
    plt.ylabel('Tip-Sample Conservative Force(nN)')
    plt.xlabel('Distance(nm)')
    plt.legend()
    plt.grid(ls = '--')
    plt.savefig(vispath+filename.split('.')[0]+'_ConsForce_L.png')
    plt.show()

    #Check whether D = z-A  is monotonically decreasing
    z_ = [z[i] - amp[i] for i in range(len(z))]
    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(z,z_, label = "Nearest point")
    plt.xlabel('distance(z)')
    plt.ylabel('neareast distance(D = z - A)')
    plt.legend()
    plt.show()
    for i in range(1,len(z_)):
        if z_[i]-z_[i-1]>0:
            continue
        else :
            print('D = z-A is not monotonically decreasing!')
            raise ValueError

    ##Force reconstruction using formula by Dagdeviren et al.
    A = Rbf(z_, amp, smooth = sm)
    ϕ = Rbf(z_, phas, smooth = sm)
    Ad = A0/Q
    def dU(D):
        return lambda z : k*(Ad*np.sin(ϕ(z+D))/A(z+D) + 1-Ω**2)*\
            (z + np.sqrt(A(z+D)*(z)/(16*np.pi)) + A(z+D)**1.5/np.sqrt(2*z))

    U = np.zeros(N)
    scheme = quadpy.line_segment.gauss_kronrod(int(500))
    for i in tqdm(range(N)):
        U[i] = scheme.integrate(dU(z_[i]), [0.0, z_[N-1] - z_[i]])
    F = -NumDiff(U,z_)

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(z,U, label = "Potential energy(Dagdeviren)", color = 'k')
    plt.xlim([0,xlim])
    plt.ylim([-15,10])
    #plt.ylim([U[np.argmin(U)]-5,U[int(xlim/(z[1]-z[0]))]+5])
    plt.ylabel('Tip-Sample Potential Energy(x$10^{-18}$J)')
    plt.xlabel('Distance(nm)')
    plt.legend()
    plt.grid(ls = '--')
    plt.savefig(vispath+filename.split('.')[0]+'_PotEnergy_D.png')
    plt.show()

    cp_D = cp0
    fit_param_DMT_D, var_DMT_D = curve_fit(lambda z, z_off, fmin  : DMT_r (z, z_off, fmin, zc = exp_param['zc'], R = exp_param['R'], H = exp_param['H'],\
         Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), z[:cp_D], F[:cp_D], p0 =[exp_param['zc']-z[cp_D], \
             -DMT(exp_param['zc'])+F[cp_D]] )

    fig, axes = plt.subplots(figsize=(8,6))
    axes.plot(z,F, label = "Conservative Force(Dagdeviren)", color = 'k')
    axes.plot(z[:cp_D],DMT_r(z[:cp_D],z_off = fit_param_DMT_D[0], fmin = fit_param_DMT_D[1], zc = exp_param['zc'], R= exp_param['R'],H = exp_param['H'],\
         Eₜ = exp_param['Eₜ'], νₜ = exp_param['νₜ'], Eₛ = exp_param['Eₛ'], νₛ = exp_param['νₛ']), label = "DMT fit", color = 'r')
    axes.axvline(x= exp_param['zc']- fit_param_DMT_D[0], linestyle = '-', color = 'r', linewidth = 5, label = 'contact point(fit) : {0:.4f}'.format(exp_param['zc']- fit_param_DMT_D[0]))
    axes.axvspan(exp_param['zc']- fit_param_DMT_D[0]-np.sqrt(np.diag(var_DMT_D))[0],exp_param['zc']- fit_param_DMT_D[0]+np.sqrt(np.diag(var_DMT_D))[0], facecolor='r', alpha=0.5)
    if 'force_params' in data:
        force_params = data["force_params"]
        axes.plot(z,DMT(z,zc = force_params['zc'], R = force_params['R'], H = force_params['H'],\
            Eₜ = force_params['Eₜ'], νₜ = force_params['νₜ'], Eₛ = force_params['Eₛ'], \
            νₛ = force_params['νₛ'], η = force_params['η']), color = 'r', ls = '--', label = 'Model Force')
        axes.axvline(x =exp_param['zc'], linestyle = '--', color = 'r',linewidth = 5, label = 'contact point(model)')

    plt.xlim([0,xlim])
    #plt.ylim([F[np.argmin(F)]-5,max(F[int(xlim/(z[1]-z[0]))]+5,F[0]+5)])
    plt.ylim([-40,100])
    plt.ylabel('Tip-Sample Conservative Force(nN)')
    plt.xlabel('Distance(nm)')
    plt.legend()
    plt.grid(ls = '--')
    plt.savefig(vispath+filename.split('.')[0]+'_ConsForce_D.png')
    plt.show()

    
    data["U_D"] = np.ndarray.tolist(U[::-1])
    data["F_D"] = np.ndarray.tolist(F[::-1])
    data["F_L"] = np.ndarray.tolist(F_k[::-1])

    with open(folderpath + filename, "w") as data_json:
        json.dump(data,data_json)
