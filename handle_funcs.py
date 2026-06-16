
import socket
import numpy as np
from scipy import interpolate as sc
import torch
import os;
import multiprocessing
import pyfftw
from pyfftw.interfaces import scipy_fftpack
import time
# pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
pi = np.pi


def slit_mean(x,n): # this is a fucntion I have defined to plot better the loss function.. it takes the slitted average of an array
    Nx = len(x) 
    N = round(np.ceil(Nx/n))
    y = np.zeros(N)
                   
    for i in range(N):
        i1 = i*n
        if (i1+n-1 > Nx-1): 
            n = Nx - i1
        y[i] = x[i1:(i1+n-1)].mean()
    return y 

def slit_mean_nd(x,n_slit=1,ind_axis=0,skip=1): # this is a fucntion I have defined to plot better the loss function.. it takes the slitted average of an array
    sx = np.array(x.shape);
    len_x = len(sx)
    Nx = sx[ind_axis]
    #skip = 1; #round(np.ceil(n_slit*0.4))
    N = round(np.ceil((Nx-n_slit )/skip));
    n2 = np.ceil(n_slit/2)
    if N == 0:
        raise ValueError('N > 0!!') 

    
   # N = round(np.ceil(Nx/n_slit))
    sy = sx+0; sy[ind_axis] = N;
    print(sx); print(sy); print(N); print(Nx); print(n_slit)
    y = np.zeros(sy)
    indy = np.zeros(sy).astype(int)
    
    # The string str_dim = ':,:,:,:,:,:' for the right dimension of the array. This string has length "2*len_x-1"
    for j in range(len_x): # Fillinf the array of string
        if j == 0:
            if j == ind_axis:
                str_dim = 'i1:(i1+n_slit-1)'; str_dimy = 'i'
            else:
                str_dim = ':'; str_dimy = ':'
        else:
            if j == ind_axis:
                str_dim = str_dim + ',i1:(i1+n_slit-1)'; str_dimy = str_dimy + ',i'
            else:
                str_dim = str_dim + ',:'; str_dimy = str_dimy + ',:'

    for i in range(N):
        i1 = i*skip
        if (i1+n_slit-1 > Nx-1): 
            n_slit = Nx - i1
            # print(    'y['+str_dimy+'] = x[' + str_dim + '].mean('+str(ind_axis)+')')
            # print(x[i1:(i1+n_slit-1),:])
            # print(n_slit)
            # print(i1,(i1+n_slit-1))
            # print(x.mean())
        exec('y['+str_dimy+'] = x[' + str_dim + '].mean('+str(ind_axis)+')')  
            
           # exec('y['+str_dimy+'] = x[' + str_dim + '].mean('+str(ind_axis)+')')  
    indy = n2-1 + skip 
    return y  

def eq_sig_int(Ik_raw,f_raw,*args):
    NL = len(args); Nk = f_raw.size; sIk = np.array(Ik_raw.shape); NsIk = sIk.size; 
    
    if not(sIk[NsIk-1] == Nk):
        print(Nk)
        print(sIk)
        raise ValueError('The last dimension of the Dataset must represent the frequencies!')
    sIx = np.array(sIk); sIx[0:NsIk-1] = 1
    f_min = f_raw[0]; f_max = f_raw[Nk-1];
    
    if NL == 0:
        Nkk = Nk
        f_vec = np.linspace(f_min,f_max,Nk);
    elif NL == 1:
        f_vec = args[0]; Nkk = f_vec.size
    # f_vec = np.array(f_vec).reshape(sIx);  
    F_interp = sc.interpolate.interp1d(f_raw, Ik_raw)
    IU = F_interp(f_vec) 
    return (IU,f_vec)
    
    
# def elasto_dataset_z(+,alpha_vec,shift_z,borders_dist,fvec,IT_CC,*args):
# # This function takes in input couples of U/L signals in k-space (A-scans,B-scans or C-scans) simulated with the Debye-Wolf integral in Matlab and creates input data for the 1D-CNN 

# #INPUT:
#         # borders -> ("+1","-1"); It specifies if the function will calculate the Fourier transform ("coef = +1") or its inverse(" coef = -1").

# IT_CC.shape

    
def gen_fft2(coef,t,F,fv,*args):
    
#This function calculates an accurate and fast approximation of the "Continuous" Fourier
#transform (or the inverse Fourier transform) by employing the built-in
#function "fft.fft()", for arbitrary sampled domain and codomain, the
#only assumption is that they are "evenly-spaced"

#DEFINITIONS:
#         Fourier transform: -->  "F(v):= Integral(f(t)*exp(-i*2*pi*t*v)*dt) "
# Inverse Fourier transform: -->  "F^(-1)(v) := Integral(f(t)*exp(+i*2*pi*t*v)*dt) "


#INPUT:
        # coef -> ("+1","-1"); It specifies if the function will calculate the Fourier transform ("coef = +1") or its inverse(" coef = -1").
        # t  -> Unidimensional array of equally spaced sampled points of the domain variable.
        # F  -> Sampled function of t. (The last dimension of F must equal the dimension of "t")
        # fv ->  -(if fv is a number -> fv = f0= "CENTRAL FREQUENCY". In this case the interval of frequencies is automatically calculated in the function)
        #       -(if fv is an interval of points, the function will evaluate the Fourier transform in the interval "fv"
        #       -(if fv is made by two numbers (fv = [fv1,fv2]), the function will evaluate the Fourier transform in the interval "np,linspace(fv1,fv2,Nf)", where Nf is the number of frequencies
        # args{1} -> It is the number of frequencies(Nf) where the Fourier transform is evaluated. It is taken in account by the function only if "fv" is composed by one number or 2 numbers. In that case, "args{1} = number  of frequencies where the Fourier transform will be evaluated. If this is missed -> "args{1} = len(t)".

#OUTPUT:
        # Fy ->  Fourier transform( or inverse Fourier transform) of the sampled array F.
        # "varargout{1}" ->  Interval of points where the Fourier transform (or its inverse) is evaluated.
        #                    If "fv" this is an array ->  "varargout{1} = fv",  if "fv" is a number -> "varargout{1}" is an interval of points centered in fv. If (fv = [fv1,fv2]) -> varargout{1} = "np,linspace(fv1,fv2,Nf)" 

#NOTE: The fastest and most accurate cases are when "fv" is a number or two numbers

############################################################################################

    Nmax = 1e7; N_acc = 5e4; #Maximum number of frequencies we let calculate by "np.fft.fft()" built_in function (More this number is large and more the approximation is accurate but slower)
    floor_eps = 1e-6; L2_cond = 0;
    
  

# checking that t is a one-dimensional array 
    ts = t.shape; Lnt = len(ts)
    if (Lnt == 1): Nt = ts[0]
    if (Lnt == 2):
        if min(ts) > 1: raise ValueError ('The domain array t must be one-dimensional')
        elif ts[0] == 1: 
            Nt = ts[1]; 
        elif ts[1] == 1: 
            Nt = ts[0]; 
        else: raise ValueError('The domain array t must be one-dimensional') #return ('The domain array t must be one-dimensional')
    t = t.reshape(Nt,1)

# checking that t is equally sampled
    # print(t.shape)
    # print(np.round(np.diff(t[:,0])*100).astype(int))
    # print((max(abs(np.diff(np.diff(t,1,0),1,0)))))
    # print(1e-8*max(abs(np.diff(t,1,0))))
    
    if (max(abs(np.diff(np.diff(t,1,0),1,0))) > 1e-8*max(abs(np.diff(t,1,0)))):
            raise ValueError('The domain has not been equally sampled')

# Checking dimensions of F and T, and that the last dimension of "F" equals the dimension of t 
    sF  = F.shape; LsF = len(sF);  
    if (sF[LsF-1] != Nt): raise ValueError('the last dimension of the dataset F must equal len(t)')
    

# Checking that the parameter coef is "+1"(FT) or "-1"(IFT)
    if ((coef != 1) and (coef != -1)):
        raise ValueError('"coef"  must be "+1" or "-1", which means that the function will calculate the Fourier transform or its inverse')

    
# Declaration of Nf in the case len(fv) = 0 or 1   
    if len(args) == 0: Nf = Nt
    elif len(args) == 1: Nf = args[0]
    else:    raise ValueError('There are too many input arguments: len(args) = {len(args)}')
           
# Declaration of the variables related to the domain
    ssT = np.ones(LsF).astype(int); ssT[LsF-1] = Nt; 
    st = np.arange(0,Nt,1).reshape(ssT).astype(float); t0 = t.mean(); 
    dt = abs(t[1]-t[0]); Mt = (Nt-1)/2;    

# case len(fv) == 1 (subcase Nf >= Nt)
    
    fv = np.array(fv); fvs = fv.size; 
    if ((fvs == 1)  and (Nf >= Nt)): # In this case fv is a number, and the interval of frequencies must be calculated inside the function
        f0 = fv; Mf = (Nf-1)/2; f_vec = np.arange(-Mf,Mf+1e-10,1)/Nf/dt + f0; ssF = ssT; ssF[LsF-1] = Nf;
        sf = np.arange(0,Nf,1).reshape(ssF).astype(float);
        Cf = np.exp(-1j*coef*2.*np.pi*(f0-Mf/dt/Nf)*st*dt);
        Cc = np.exp(-1j*coef*2*np.pi*(t0-Mt*dt)*(f0+(sf-Mf)/dt/Nf));
        
        if coef == 1:
            Fy = np.fft.fft(F*Cf,Nf)*Cc*dt
        else:
            Fy = np.fft.ifft(F*Cf,Nf)*Nf*Cc*dt 
        return Fy, f_vec
    else: 
 # case len(fv) == 1 (subcase Nf < Nt.. zeropadding). In conjunction with the previous "if", this statement implies even that "Nf < Nt", and in this case the array of frequencies "fv" is calculated below.
        if (fv.size == 1):   Mf = (Nf-1)/2; f0 = fv; fv = np.arange(-Mf,Mf+1e-10,1)/Nf/dt + f0;
        elif (fv.size == 2): #The array of frequencies  is is bounded by the two frequencies [fv1,fv2]
            f_min = fv[0]; f_max = fv[1]; L2_cond = 1;
            if (len(args) == 2):  # it means that varargin(2) is the number of zero padding
                Np2 = args[1]; # zero-padding (chosen)
                fv = np.arange(f_min,f_max+1e-10,1/dt/Np2);  Nf = len(fv); Mf = (Nf-1)/2;
            else: Np2 = np.round((Nf-1)/dt/(fv[1]-fv[0])); f_max2 = f_min + 1/dt/Np2*(Nf-1); fv = np.arange(f_min,f_max2+1e-10,1/dt/Np2); Nf = fv.size; Mf = (Nf-1)/2;
        else:  
             raise ValueError('len(fv) > 2!'); # The input of frequencies fv is an array of "Nf" components (Nf is an input)
             Nf = fv.size; Mf = (Nf-1)/2;
             if (max(abs(np.diff(np.diff(fv,0,1),0,1))) > 1e-10*max(abs(np.diff(fv,0,1),0,1))):
                 raise ValueError('The codomain has not been equally sampled')
        f0 = fv.mean();  
        if (L2_cond): df = 1/dt/Np2
        else: df = abs(fv[1]-fv[0]) 
        
# Calculation of the "right" zero-padding estimating p,q s.t: df*dt = p/q. In this case the denominator q is the right number of zero padding.
        x = df*dt; q1 = np.round(1/x);  err_toll2 = abs(df-1/q1/dt); q = q1;
        if ((err_toll2 < 1e-7) and (q < Nmax)):  #in that case df ≈ 1/q/dt
            p1 = 1;
            if (q < Nt): # In this case I find a suitable multiple of "q" that is greater than Nt
                qq = Nt-np.mod(Nt,q)+q  # qq = q*Nq (multiple)
                p1 = np.round(p1/q*qq+floor_eps).astype(int); p1 = p1[0]  #p1 s.t. p1/q1 = p/q
                q = np.round(qq +floor_eps).astype(int); 
            Nf1 = round(q[0].astype(int)); Mf1 = (Nf1-1)/2; #f01 = f0 + 1/dt*Mf1/Nf1- Mf*df;
            tmp = 0; NftM = 0; Fy = np.zeros([*sF[0:(LsF-1)],Nf],dtype = complex);
            while (tmp == 0):
                Nftm = NftM+1;
                f01 = f0 + (Mf1)/dt/Nf1 - (Mf-NftM)/dt/q1; #fvec1 = [-Mf1:Mf1]/dt/Nf1 + f01;
                ssF1 = ssT; ssF1[LsF-1] = Nf1; sf1 = np.arange(0,Nf1,1).reshape(ssF1).astype(float);
                Cf1 = np.exp(-1j*coef*2*np.pi*(f01-Mf1/dt/Nf1)*st*dt)
                Cc1 = np.exp(-1j*coef*2*np.pi*(t0-Mt*dt)*(f01+(sf1-Mf1)/dt/Nf1))
                
                if coef == 1:
                    Fy1 = np.fft.fft(F*Cf1,Nf1)*Cc1*dt
                else:
                    Fy1 = np.fft.ifft(F*Cf1,Nf1)*Nf1*Cc1*dt 
                
                Nft1 = np.floor((Nf1-1)/p1 + Nftm + floor_eps); NftM = Nft1;
                if (NftM >= Nf): NftM = Nf; tmp = 1;
                for j1 in np.arange(Nftm,NftM+1e-10,1).astype(int):
                    jp = p1*(j1-Nftm)+1; jp = jp.astype(int)
                    if LsF == 1: Fy[j1-1] = Fy1[jp-1]
                    elif LsF == 2: Fy[:,j1-1] = Fy1[:,jp-1]
                    elif LsF == 3: Fy[:,:,j1-1] = Fy1[:,:,jp-1]
                    elif LsF == 4: Fy[:,:,:,j1-1] = Fy1[:,:,:,jp-1] 
            return (Fy,fv)
        else: raise ValueError('1/df/dt is not integer! Probably this means that len(fv) > 2')
        
def gen_fftW(coef,t,F,fv,*args):
    
#This function calculates an accurate and fast approximation of the "Continuous" Fourier
#transform (or the inverse Fourier transform) by employing the built-in
#function "fft.fft()", for arbitrary sampled domain and codomain, the
#only assumption is that they are "evenly-spaced"

#DEFINITIONS:
#         Fourier transform: -->  "F(v):= Integral(f(t)*exp(-i*2*pi*t*v)*dt) "
# Inverse Fourier transform: -->  "F^(-1)(v) := Integral(f(t)*exp(+i*2*pi*t*v)*dt) "


#INPUT:
        # coef -> ("+1","-1"); It specifies if the function will calculate the Fourier transform ("coef = +1") or its inverse(" coef = -1").
        # t  -> Unidimensional array of equally spaced sampled points of the domain variable.
        # F  -> Sampled function of t. (The last dimension of F must equal the dimension of "t")
        # fv ->  -(if fv is a number -> fv = f0= "CENTRAL FREQUENCY". In this case the interval of frequencies is automatically calculated in the function)
        #       -(if fv is an interval of points, the function will evaluate the Fourier transform in the interval "fv"
        #       -(if fv is made by two numbers (fv = [fv1,fv2]), the function will evaluate the Fourier transform in the interval "np,linspace(fv1,fv2,Nf)", where Nf is the number of frequencies
        # args{1} -> It is the number of frequencies(Nf) where the Fourier transform is evaluated. It is taken in account by the function only if "fv" is composed by one number or 2 numbers. In that case, "args{1} = number  of frequencies where the Fourier transform will be evaluated. If this is missed -> "args{1} = len(t)".

#OUTPUT:
        # Fy ->  Fourier transform( or inverse Fourier transform) of the sampled array F.
        # "varargout{1}" ->  Interval of points where the Fourier transform (or its inverse) is evaluated.
        #                    If "fv" this is an array ->  "varargout{1} = fv",  if "fv" is a number -> "varargout{1}" is an interval of points centered in fv. If (fv = [fv1,fv2]) -> varargout{1} = "np,linspace(fv1,fv2,Nf)" 

#NOTE: The fastest and most accurate cases are when "fv" is a number or two numbers

############################################################################################
    
    Nmax = 1e7; N_acc = 5e4; #Maximum number of frequencies we let calculate by "np.fft.fft()" built_in function (More this number is large and more the approximation is accurate but slower)
    floor_eps = 1e-6; L2_cond = 0;
    
    # import multiprocessing
    # import pyfftw
    # from pyfftw.interfaces import scipy_fftpack
    # import time
    # pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
    
# checking that t is a one-dimensional array 
    ts = t.shape; Lnt = len(ts)
    if (Lnt == 1): Nt = ts[0]
    if (Lnt == 2):
        if min(ts) > 1: raise ValueError ('The domain array t must be one-dimensional')
        elif ts[0] == 1: 
            Nt = ts[1]; 
        elif ts[1] == 1: 
            Nt = ts[0]; 
        else: raise ValueError('The domain array t must be one-dimensional') #return ('The domain array t must be one-dimensional')
    t = t.reshape(Nt,1)

# checking that t is equally sampled
    if (max(abs(np.diff(np.diff(t,1,0),1,0))) > 1e-8*max(abs(np.diff(t,1,0)))):
            raise ValueError('The domain has not been equally sampled')

# Checking dimensions of F and T, and that the last dimension of "F" equals the dimension of t 
    sF  = F.shape; LsF = len(sF);  
    if (sF[LsF-1] != Nt): raise ValueError('the last dimension of the dataset F must equal len(t)')
    

# Checking that the parameter coef is "+1"(FT) or "-1"(IFT)
    if ((coef != 1) and (coef != -1)):
        raise ValueError('"coef"  must be "+1" or "-1", which means that the function will calculate the Fourier transform or its inverse')

    
# Declaration of Nf in the case len(fv) = 0 or 1   
    if len(args) == 0: Nf = Nt
    elif len(args) == 1: Nf = args[0]
    else:    raise ValueError('There are too many input arguments: len(args) = {len(args)}')
           
# Declaration of the variables related to the domain
    ssT = np.ones(LsF).astype(int); ssT[LsF-1] = Nt; 
    st = np.arange(0,Nt,1).reshape(ssT).astype(float); t0 = t.mean(); 
    dt = abs(t[1]-t[0]); Mt = (Nt-1)/2;    

# case len(fv) == 1 (subcase Nf >= Nt)
    
    fv = np.array(fv); fvs = fv.size; 
    if ((fvs == 1)  and (Nf >= Nt)): # In this case fv is a number, and the interval of frequencies must be calculated inside the function
        f0 = fv; Mf = (Nf-1)/2; f_vec = np.arange(-Mf,Mf+1e-10,1)/Nf/dt + f0; ssF = ssT; ssF[LsF-1] = Nf;
        sf = np.arange(0,Nf,1).reshape(ssF).astype(float);
        Cf = np.exp(-1j*coef*2.*np.pi*(f0-Mf/dt/Nf)*st*dt);
        Cc = np.exp(-1j*coef*2*np.pi*(t0-Mt*dt)*(f0+(sf-Mf)/dt/Nf));
        
        if coef == 1:
            # Fy = np.fft.fft(F*Cf,Nf)*Cc*dt
            Fy = scipy_fftpack.fft(F*Cf,Nf)*Cc*dt
        else:
            # Fy = np.fft.ifft(F*Cf,Nf)*Nf*Cc*dt 
            Fy = scipy_fftpack.ifft(F*Cf,Nf)*Nf*Cc*dt 
        return Fy, f_vec
    else: 
 # case len(fv) == 1 (subcase Nf < Nt.. zeropadding). In conjunction with the previous "if", this statement implies even that "Nf < Nt", and in this case the array of frequencies "fv" is calculated below.
        if (fv.size == 1):   Mf = (Nf-1)/2; f0 = fv; fv = np.arange(-Mf,Mf+1e-10,1)/Nf/dt + f0;
        elif (fv.size == 2): #The array of frequencies  is is bounded by the two frequencies [fv1,fv2]
            f_min = fv[0]; f_max = fv[1]; L2_cond = 1;
            if (len(args) == 2):  # it means that varargin(2) is the number of zero padding
                Np2 = args[1]; # zero-padding (chosen)
                fv = np.arange(f_min,f_max+1e-10,1/dt/Np2);  Nf = len(fv); Mf = (Nf-1)/2;
            else: Np2 = np.round((Nf-1)/dt/(fv[1]-fv[0])); f_max2 = f_min + 1/dt/Np2*(Nf-1); fv = np.arange(f_min,f_max2+1e-10,1/dt/Np2); Nf = fv.size; Mf = (Nf-1)/2;
        else:  
             raise ValueError('len(fv) > 2!'); # The input of frequencies fv is an array of "Nf" components (Nf is an input)
             Nf = fv.size; Mf = (Nf-1)/2;
             if (max(abs(np.diff(np.diff(fv,0,1),0,1))) > 1e-10*max(abs(np.diff(fv,0,1),0,1))):
                 raise ValueError('The codomain has not been equally sampled')
        f0 = fv.mean();  
        if (L2_cond): df = 1/dt/Np2
        else: df = abs(fv[1]-fv[0]) 
        
# Calculation of the "right" zero-padding estimating p,q s.t: df*dt = p/q. In this case the denominator q is the right number of zero padding.
        x = df*dt; q1 = np.round(1/x);  err_toll2 = abs(df-1/q1/dt); q = q1;
        if ((err_toll2 < 1e-7) and (q < Nmax)):  #in that case df ≈ 1/q/dt
            p1 = 1;
            if (q < Nt): # In this case I find a suitable multiple of "q" that is greater than Nt
                qq = Nt-np.mod(Nt,q)+q  # qq = q*Nq (multiple)
                p1 = np.round(p1/q*qq+floor_eps).astype(int); p1 = p1[0]  #p1 s.t. p1/q1 = p/q
                q = np.round(qq +floor_eps).astype(int); 
            Nf1 = round(q[0].astype(int)); Mf1 = (Nf1-1)/2; #f01 = f0 + 1/dt*Mf1/Nf1- Mf*df;
            tmp = 0; NftM = 0; Fy = np.zeros([*sF[0:(LsF-1)],Nf],dtype = complex);
            while (tmp == 0):
                Nftm = NftM+1;
                f01 = f0 + (Mf1)/dt/Nf1 - (Mf-NftM)/dt/q1; #fvec1 = [-Mf1:Mf1]/dt/Nf1 + f01;
                ssF1 = ssT; ssF1[LsF-1] = Nf1; sf1 = np.arange(0,Nf1,1).reshape(ssF1).astype(float);
                Cf1 = np.exp(-1j*coef*2*np.pi*(f01-Mf1/dt/Nf1)*st*dt)
                Cc1 = np.exp(-1j*coef*2*np.pi*(t0-Mt*dt)*(f01+(sf1-Mf1)/dt/Nf1))
            
                # timetest = time.time()
                if coef == 1:
                    # Fy1 = np.fft.fft(F*Cf1,Nf1)*Cc1*dt
                    Fy1 = scipy_fftpack.fft(F*Cf1,Nf1)*Cc1*dt
                else:
                    # Fy1 = np.fft.ifft(F*Cf1,Nf1)*Nf1*Cc1*dt 
                    Fy1 = scipy_fftpack.ifft(F*Cf1,Nf1)*Nf1*Cc1*dt 
                # print(f'padding = {Nf1}')
                # print(f'total length time is : {time.time()-timetest}')
                Nft1 = np.floor((Nf1-1)/p1 + Nftm + floor_eps); NftM = Nft1;
                if (NftM >= Nf): NftM = Nf; tmp = 1;
                for j1 in np.arange(Nftm,NftM+1e-10,1).astype(int):
                    jp = p1*(j1-Nftm)+1; jp = jp.astype(int)
                    if LsF == 1: Fy[j1-1] = Fy1[jp-1]
                    elif LsF == 2: Fy[:,j1-1] = Fy1[:,jp-1]
                    elif LsF == 3: Fy[:,:,j1-1] = Fy1[:,:,jp-1]
                    elif LsF == 4: Fy[:,:,:,j1-1] = Fy1[:,:,:,jp-1] 
         #       import torch; print(f'inside gen_fft. Memory allocated after CNN {torch.cuda.memory_allocated()},Max Memory {torch.cuda.max_memory_allocated()}')
             
            return (Fy,fv)
        else: raise ValueError('1/df/dt is not integer! Probably this means that len(fv) > 2')
        
def ff_tr(refind,It,F,freq):
    #breakpoint()
    Nt = len(It); DT = np.zeros([Nt,1]); DT[0] = It[1]-It[0];
    for it in range(Nt-2):
        DT[it+1] = (It[it+2]-It[it])/2; #DT[it]/2 + (It[it+2]-It[it+1])/2;
    DT[Nt-1] = It[Nt-1]-It[Nt-2];
    
    Nk = len(freq)
    sx = F.shape; Lx = len(np.squeeze(sx)); # print(Lx)
    g = np.zeros(np.append(sx[0:Lx-1],Nk),dtype=complex)
    if (Lx == 1 ) or ( Lx == 2 and (sx[0] == 1)):
        for k in range(Nk):
            if Nt < 3:
                for j in range(Nt):
                    dt = DT[j]; g[k] = g[k] + F[j]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j],2*pi))*dt
            else: 
                for j in range(Nt-2):
                    dt = DT[j+1]; g[k] = g[k] + F[j+1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j+1],2*pi))*dt
                g[k] = g[k] + F[0]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[0],2*pi))*DT[0]/2 + F[Nt-1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[Nt-1],2*pi))*DT[Nt-1]/2;
        return g
    elif (Lx == 2):
        for k in range(Nk):
            if Nt < 3:
                for j in range(Nt):
                    dt = DT[j+1]; g[:,k] = g[:,k] + F[:,j]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j],2*pi))*dt
            else: 
                for j in range(Nt-2):
                    dt = DT[j+1]; g[:,k] = g[:,k] + F[:,j+1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j+1],2*pi))*dt
                g[:,k] = g[:,k] + F[:,0]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[0],2*pi))*DT[0]/2 + F[:,Nt-1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[Nt-1],2*pi))*DT[Nt-1]/2;
        return g
    elif (Lx == 3): 
        for k in range(Nk):
            if Nt < 3:
                for j in range(Nt):
                    dt = DT[j+1]; g[:,:,k] = g[:,:,k] + F[:,:,j]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j],2*pi))*dt
            else: 
                for j in range(Nt-2):
                    dt = DT[j+1]; g[:,:,k] = g[:,:,k] + F[:,:,j+1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j+1],2*pi))*dt
                g[:,:,k] = g[:,:,k] + F[:,:,0]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[0],2*pi))*DT[0]/2 + F[:,:,Nt-1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[Nt-1],2*pi))*DT[Nt-1]/2
        #import torch; print(f'inside ff_tr. Memory allocated after CNN {torch.cuda.memory_allocated()},Max Memory {torch.cuda.max_memory_allocated()}')
        return g
    elif (Lx == 4): 
        for k in range(Nk):
            if Nt < 3:
                for j in range(Nt):
                    dt = DT[j+1]; g[:,:,:,k] = g[:,:,:,k] + F[:,:,:,j]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j],2*pi))*dt
            else: 
                for j in range(Nt-2):
                    dt = DT[j+1]; g[:,:,:,k] = g[:,:,:,k] + F[:,:,:,j+1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[j+1],2*pi))*dt
                g[:,:,:,k] = g[:,:,:,k] + F[:,:,:,0]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[0],2*pi))*DT[0]/2 + F[:,:,:,Nt-1]*np.exp(-1j*np.mod(2*pi*freq[k]*refind*It[Nt-1],2*pi))*DT[Nt-1]/2
        return g
    else:
        raise ValueError('Dimensioon too high!') 

def err_n(v1,v2):
    if torch.is_tensor(v1) and torch.is_tensor(v2):
        sv1 = v1.size(); sv2 = v2.size(); 
        # print(sv1); print(sv2)
        if not(sv1==sv2):
            raise ValueError('The arrays have different dimensions!') 
        
        v1 = v1.flatten(); v2 = v2.flatten();
        err_v = 2*torch.norm(v1-v2)/(torch.norm(v1)+torch.norm(v2))
    else: 
        sv1 = v1.shape; sv2 = v2.shape;  Lv = len(sv1);
        if not(sv1==sv2):
            raise ValueError('The arrays have different dimensions!') 
        v1 = v1.flatten(); v2 = v2.flatten();
        err_v = 2*np.linalg.norm(v1-v2)/(np.linalg.norm(v1)+np.linalg.norm(v2))
    return err_v


def sub_inds(A,*args): # function that allows to pass arrys of indices to the variable; Example: B = sub_inds(A,I1,I2,I3,I4) is the equivalent of the matlab code: B = A(I1,I2,I3,I4)
    
    num_arg = len(args)
    sa = A.shape; La = len(sa); rem_inds = tuple(np.arange(La,7)); saL = np.zeros(7).astype(int); saL[0:La] = sa;
    for ir in range(len(rem_inds)):
        saL[ir+La] = 1
    A_L = np.reshape(A,saL)
    if num_arg>La: 
        raise ValueError('The number of incides exceeds the array dimension')
    if num_arg==0: 
        raise ValueError('The number of inputs must be greater than 1!')
    
    IT = {}; zero = np.array([0])
    IT['0'] = zero; IT['1'] = zero; IT['2'] = zero; IT['3'] = zero; IT['4'] = zero; IT['5'] = zero; IT['6'] = zero; IT['7'] = zero;
    for ia in range(La):
        sia = str(ia)
        IT[sia] = np.arange(sa[ia]);    
        
    for ian in range(num_arg):
        sian = str(ian)
        if np.max(args[ian]) > sa[ian]: 
            raise ValueError(f'The indices {ia} is greater than the related dimension')
        IT[sian] = args[ian] 
        
    B = A_L[IT['0'][:, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis],IT['1'][np.newaxis, :,np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis], IT['2'][np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis],IT['3'][ np.newaxis, np.newaxis, np.newaxis, :,np.newaxis, np.newaxis, np.newaxis],IT['4'][ np.newaxis, np.newaxis, np.newaxis, np.newaxis,:, np.newaxis, np.newaxis],IT['5'][ np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis,:, np.newaxis],IT['6'][ np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]]
    
    return np.squeeze(B, axis=rem_inds)

def init_paths(str_saving):
    desk = '/home/andrea/MATLAB_26_04_2022/Matlab_drive_scripts/'
    mnem = '/home/amazzolani/Matlab_drive_scripts/'
    zeus = '/home/amazzolani/Matlab_drive_scripts/'
    clio = '/home/amazzolani/Matlab_drive_scripts/'
    NWW  = 'C:/Users/xpci2021/Documents/MATLAB/Andrea/Matlab_drive_scripts/'
    myri = '/home/rmapaam/Scratch/Matlab_drive_scripts/'
    lapt = 'C:/Users/andre/Documents/MATLAB/Matlab_drive_scripts/'
    srs = '/home/amazzolani/Matlab_drive_scripts/'
    
    desk_p = 'C:/Users/andre/spyder-folder/spyder_py3/'
    mnem_p = '/home/amazzolani/spyder-folder/spyder_py3/'
    zeus_p = '/home/amazzolani/spyder-folder/spyder_py3/'
    clio_p = '/home/amazzolani/spyder-folder/spyder_py3/'
    NWW_p  = 'C:/Users/xpci2021/Documents/MATLAB/Andrea/spyder_py3/'
    myri_p  = '/home/rmapaam/Scratch/spyder-folder/spyder_py3/'
    lapt_p  = 'C:/Users/andre/spyder-folder/spyder_py3/'
    srs_p  = '/home/amazzolani/spyder-folder/spyder_py3/'

    desk_NAS = '/home/andrea/MATLAB_26_04_2022/Matlab_drive_scripts/remotedata_Andrea/'
    mnem_NAS = '/home/amazzolani/Matlab_drive_scripts/remotedata/'
    zeus_NAS = '/home/amazzolani/Matlab_drive_scripts/remotedata/'
    clio_NAS = '/home/amazzolani/Matlab_drive_scripts/remotedata/'
    NWW_NAS  = 'Z:/'
    myri_NAS  = 'nofolder'
    lapt_NAS  = 'Z:/'
    srs_NAS  = '/home/amazzolani/Matlab_drive_scripts/remote_folder_Andrea/'

    host_list = ["dylan-Precision-5820-Tower-X-Series","Mnemosyne","Zeus","Clio","DESKTOP-19KUFTV","login13.myriad.ucl.ac.uk","DESKTOP-O4UR6DD","asusrs720a"]    
    host_IP_list = ["128.40.160.198","128.40.68.241","128.40.68.242","128.40.68.244","128.40.160.78","192.168.122.1","144.82.114.238","128.40.68.245"]

    hostname = socket.gethostname(); # print(hostname)
    ip_address = socket.gethostbyname(hostname); # print(ip_address)

    location_mat = [desk,mnem,zeus,clio,NWW,myri,lapt,srs]
    location_pyt = [desk_p,mnem_p,zeus_p,clio_p,NWW_p,myri_p,lapt_p,srs_p]
    location_NAS = [desk_NAS,mnem_NAS,zeus_NAS,clio_NAS,NWW_NAS,myri_NAS,lapt_NAS,srs_NAS]
    NL = len(host_list)
    path_str_mat = 1    
    for it in range(NL):
        path_test_mat = location_mat[it]
        path_test_pyt = location_pyt[it]
        path_test_NAS = location_NAS[it]
        
        if hostname==host_list[it]:
            host = host_list[it] 
            path_str_mat = path_test_mat  
            path_str_pyt = path_test_pyt  
            path_str_sav = path_test_mat + str_saving
            path_str_NAS = path_test_NAS
    if path_str_mat == 1: raise ValueError('The path folder does not exist!')
        
    return path_str_mat,path_str_pyt,path_str_sav, host,path_str_NAS
 
def ndzeros(N_vec,*args):
    N_tot = np.prod(N_vec);
    
    if len(args) == 0: A = np.zeros(N_tot); A = A.reshape(N_vec)
    elif (len(args) == 1):
        if (args[0] == 'complex' ) or (args[0] == 'Complex' ) or (args[0] == 'c' ) or (args[0] == 'comp' ) or (args[0] == 'Comp' ): 
            A = np.zeros(N_tot,dtype=complex); A = A.reshape(N_vec)
        elif (args[0] == 'real' ) or (args[0] == 'Real' ) or (args[0] == 'r' ) or (args[0] == 'real' ): 
            A = np.zeros(N_tot); A = A.reshape(N_vec)
        else:    raise ValueError('The second argument must be one of those strings: "complex" or "real"')
    else:    raise ValueError('There are too many input arguments: len(args) = {len(args)}')
    
    return A
    
def prec_set(x,int_prec,*args): # this function stores an approximation of the array in an array of uint8, uint16 or uint32, by reducing the amount of memory employed 
    
    vec_prec = [8,16,32,64]; int_str = '';  Lv = len(vec_prec)
    for it in range(Lv):
        if int_prec == vec_prec[it]:
            int_str = str(int_prec); 

    sx = x.shape; Lx  = len(sx);
    if len(int_str) == 0: 
        raise ValueError('The second input must be an integer among "8", "16", "32", "64"')
    
    sub_ind = np.arange(Lx); num_var = len(args); 
    if num_var == 1:
        sub_ind = tuple(args[0])
    if num_var == 2:
        raise ValueError('Too many input')
    sub_ind = tuple(sub_ind)
    min_x = np.min(x,axis=sub_ind,keepdims=True);
    max_x = np.max(x,axis=sub_ind,keepdims=True);
    D_x = max_x-min_x; D_u = 2**int_prec - 1; D_X = D_x*np.ones(sx)
    
    # print('x_prec = '+str_int+'(D_u*(x-min_x)/D_x);')       
    # x_prec = exec(str_int+'(D_u*(x-min_x)/D_x);'); 
    xf = np.round(D_u*(x-min_x)/D_X)
    if int_prec == 8:
        x_prec = np.uint8(xf);
    if int_prec == 16:
        x_prec = np.uint16(xf);
    if int_prec == 32:
        x_prec = np.uint32(xf);
    if int_prec == 64:
        x_prec = np.uint64(xf);
    
    # print(x_prec.shape)
    x_prec[D_X == 0] = 0;
    min_x = np.array(min_x).astype('float64')
    D_x = np.array(D_x).astype('float64')
    D_u = np.array(D_u).astype('float64')
    return [x_prec,min_x,D_x,D_u]

def prec_inv(x_cell,*args):
# It is linked to "prec_set" function, and it restores the "double" precision estimation from "uint"    
    if isinstance(x_cell[0],list):
        Lx_tmp = len(x_cell[0][0].shape)
        x_cell[0] = np.array(x_cell[0][0])
        if np.array([x_cell[1]]).shape == (1,1):
            for ix in range(Lx_tmp-1):
                x_cell[1] = np.array([x_cell[1]])
                x_cell[2] = np.array([x_cell[2]])
                x_cell[3] = np.array([x_cell[3]])
    x = x_cell[0] ; sx = x.shape; Lx = len(sx);  
    # print(f'shapes = {x_cell[0].dtype,x_cell[1].dtype,x_cell[2].dtype,x_cell[3].dtype}')
    D_x = x_cell[2]; sdx = D_x.shape; Ldx = len(sdx); 
    Indx_T = {}; Lx_T = {}; IndDx_T = {}; Ldx_T = {};
    for ix in range(Lx):
        six = str(ix); Indx_T[six] = np.arange(sx[ix]);  Lx_T[six] = len(Indx_T[six])
    for idx in range(Ldx):
        sidx = str(idx); IndDx_T[sidx] = np.arange(sdx[idx]);  Ldx_T[sidx] = len(IndDx_T[sidx])
    num_var = len(args)
    # print(Indx_T.keys())
    iarg = 0;
    while num_var > iarg:
        sia = str(iarg); Indx_T[sia] = args[iarg]; Lx_T[sia] = len(Indx_T[sia]); iarg = iarg+1;
    iarg2 = 0;
    while (num_var > iarg2) and (iarg2 < Ldx):
        sia2 = str(iarg2); 
        if not(np.array_equal(IndDx_T[sia2],[0])):
               IndDx_T[sia2] = args[iarg2]; Lx_T[sia2] = len(IndDx_T[sia2]); 
        iarg2 = iarg2 + 1;

    
    str_indx = '[Ind_0[0]:Ind_0[Lx_0-1]+1'; 
    for ix in range(Lx-1): 
        six = str(ix+1)
        str_indx = str_indx + ',Ind_'+six + '[0]:Ind_' + six + '[Lx_' + six + ' -1] + 1'
    str_indx = str_indx + ']';
    
    str_indD = '[Ind_D0[0]:Ind_D0[Ldx_0-1]+1'; 
    for idx in range(Ldx-1): 
        sidx = str(idx+1)
        str_indD = str_indD + ',Ind_D'+sidx + '[0]:Ind_D' + sidx + '[Ldx_' + sidx + ' -1] + 1'
    str_indD = str_indD + ']';
    
    # y0 = x_cell[1][Indx_T['0'],Indx_T['1'],Indx_T['2'],Indx_T['3'],Indx_T['4'],Indx_T['0']]
    x1 = x_cell[1]; 
    if Lx == 1: p1 = sub_inds(x1,IndDx_T['0']); p2 = np.double(sub_inds(x,Indx_T['0'])); p3 = sub_inds(D_x,IndDx_T['0']);
    if Lx == 2: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1']);
    if Lx == 3: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1'],IndDx_T['2']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'],Indx_T['2'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1'],IndDx_T['2']);
    if Lx == 4: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'],Indx_T['2'],Indx_T['3'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3']);
    if Lx == 5: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'],Indx_T['2'],Indx_T['3'],Indx_T['4'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4']);
    if Lx == 6: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4'],IndDx_T['5']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'],Indx_T['2'],Indx_T['3'],Indx_T['4'],Indx_T['5'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4'],IndDx_T['5']);
    if Lx == 7: p1 = sub_inds(x1,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4'],IndDx_T['5'],IndDx_T['6']); p2 = np.double(sub_inds(x,Indx_T['0'],Indx_T['1'],Indx_T['2'],Indx_T['3'],Indx_T['4'],Indx_T['5'],Indx_T['6'])); p3 = sub_inds(D_x,IndDx_T['0'],IndDx_T['1'],IndDx_T['2'],IndDx_T['3'],IndDx_T['4'],IndDx_T['5'],IndDx_T['6']);
    # print(f'Lx = {Lx}')
    return p1 + p2*p3/x_cell[3];
# Test prec_set --> prec_inv
# A = np.random.rand(10,51,17,18,5,3,4); A[5,:,:,15,:,:,:] = 0; A_prec_cell = prec_set(A,8,[1,2]);
# I1 = np.arange(3,7); I2 = np.arange(20,40); I3 = np.arange(1,5); I4 = np.arange(13,17);
# A_est = prec_inv(A_prec_cell,I1,I2,I3,I4); err_n(sub_inds(A,I1,I2,I3,I4),A_est)

def prec_set2(x, int_prec, *axes):
    """
    Compress an array x by approximating it with an integer representation 
    of the specified bit precision (8, 16, 32, or 64).
    
    Parameters:
        x         : Input NumPy array.
        int_prec  : Integer precision in bits; must be one of 8, 16, 32, or 64.
        axes      : Optional. If provided, should be a single list/tuple of axis indices 
                    along which to compute the min and max. If omitted, the min and max 
                    are computed over all axes.
                    
    Returns:
        A list [x_prec, min_x, D_x, D_u] where:
          - x_prec : The compressed (quantized) array, stored in the specified unsigned type.
          - min_x  : The minimum value(s) computed along the given axes (with keepdims=True).
          - D_x    : The difference (max - min) along the given axes.
          - D_u    : The maximum representable integer (2**int_prec - 1).
    """
    # Validate int_prec
    if int_prec not in (8, 16, 32, 64):
        raise ValueError('int_prec must be one of 8, 16, 32, or 64')
    
    # Determine axes along which to compute min and max.
    if len(axes) == 0:
        axes = tuple(range(x.ndim))
    elif len(axes) == 1:
        # allow user to pass a single tuple/list of axes
        if isinstance(axes[0], (list, tuple, np.ndarray)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)
    else:
        raise ValueError('Too many input axes')
    
    # Compute min and max (with keepdims for proper broadcasting later)
    min_x = np.min(x, axis=axes, keepdims=True)
    max_x = np.max(x, axis=axes, keepdims=True)
    D_x = max_x - min_x
    D_u = 2**int_prec - 1
    
    # Compute quantization scale, taking care to avoid division by zero.
    # When D_x is zero (constant data), the scale is set to 0.
    scale = np.where(D_x == 0, 0, D_u / D_x)
    xf = np.round((x - min_x) * scale)
    
    # Convert to the appropriate unsigned integer type.
    if int_prec == 8:
        x_prec = xf.astype(np.uint8)
    elif int_prec == 16:
        x_prec = xf.astype(np.uint16)
    elif int_prec == 32:
        x_prec = xf.astype(np.uint32)
    elif int_prec == 64:
        x_prec = xf.astype(np.uint64)
    
    # In regions where D_x == 0, force quantized values to zero.
    x_prec = np.where(D_x == 0, 0, x_prec)
    
    return [x_prec, min_x.astype(np.float64), D_x.astype(np.float64), float(D_u)]

def prec_inv2(x_cell, *indices):
    """
    Reconstruct the original (approximate) array from its compressed representation.
    
    Parameters:
        x_cell  : List returned by prec_set: [x_prec, min_x, D_x, D_u].
        indices : Optional. One or more 1D arrays (one per dimension of min_x/D_x)
                  used to subset the stored arrays. For example, if min_x has shape (1, 20)
                  and you want to retrieve a subset along the second axis, provide an index array
                  for that axis.
                  
    Returns:
        The reconstructed array (as float64), computed by:
              min_x + (x_prec * D_x / D_u)
        If indices are provided, the operation is performed on the subset of each array.
    """
    x_prec, min_x, D_x, D_u = x_cell
    
    # If indices are provided, use np.ix_ for multi-dimensional subsetting.
    if indices:
        x_prec = x_prec[np.ix_(*indices)]
        min_x  = min_x[np.ix_(*indices)]
        D_x    = D_x[np.ix_(*indices)]
    
    # Reconstruct the array.
    return min_x + x_prec.astype(np.float64) * D_x / D_u