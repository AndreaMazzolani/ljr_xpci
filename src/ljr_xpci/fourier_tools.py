import numpy as np


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
def gen_fft2(transform_sign, sample_positions, sampled_values, output_positions, *args):
    """
    Evaluate a continuous Fourier transform or inverse transform along the last axis.

    Parameters
    ----------
    transform_sign : int
        Use +1 for the transform with kernel exp(-1j*2*pi*x*f). Use -1 for the inverse transform with kernel exp(+1j*2*pi*x*f).

    sample_positions : np.ndarray, shape (Nx,)
        Equally spaced coordinates of the sampled axis. This is the coordinate vector associated with sampled_values[..., :].

    sampled_values : np.ndarray, shape (..., Nx)
        Values sampled on sample_positions. The last dimension must have length Nx. Any leading dimensions are treated as independent batch dimensions.

    output_positions : float or np.ndarray
        Coordinates where the transform is evaluated. If scalar, an equally spaced output grid centred on this value is generated. If shape (Nout,), the transform is evaluated directly on those coordinates.

    *args : optional
        If one value is provided and output_positions is scalar, it sets the number of output samples Nout. If omitted, Nout = Nx.

    Returns
    -------
    transformed_values : np.ndarray, shape (..., Nout)
        Fourier-transformed values evaluated on output_grid.

    output_grid : np.ndarray, shape (Nout,)
        Coordinates where transformed_values is evaluated.
    """

    sample_positions = np.asarray(sample_positions, dtype=float).reshape(-1); sampled_values = np.asarray(sampled_values)  # input coordinates and data
    nx = sample_positions.size; n_args = len(args)  # number of samples and optional inputs

    if sampled_values.shape[-1] != nx: raise ValueError("The last dimension of sampled_values must match len(sample_positions).")
    if transform_sign not in (1, -1): raise ValueError("transform_sign must be +1 or -1.")
    if nx < 2: raise ValueError("sample_positions must contain at least two samples.")
    if n_args > 1: raise ValueError("Only one optional argument is supported: the number of output samples.")

    dx = float(np.mean(np.diff(sample_positions)))  # sampling step along the input axis
    if not np.allclose(np.diff(sample_positions), dx, rtol=1e-7, atol=1e-12): raise ValueError("sample_positions must be equally spaced.")

    output_positions = np.asarray(output_positions, dtype=float).reshape(-1)  # requested output coordinates
    if output_positions.size == 1:
        nout = int(args[0]) if n_args == 1 else nx; output_grid = (np.arange(nout, dtype=float) - (nout - 1) / 2.0) / (nout * abs(dx)) + float(output_positions[0])
    elif output_positions.size == 2 and n_args == 1:
        nout = int(args[0]); output_grid = np.linspace(output_positions[0], output_positions[1], nout)
    else:
        output_grid = output_positions; nout = output_grid.size
    phase_matrix = np.exp(-1j * transform_sign * 2.0 * np.pi * np.outer(sample_positions, output_grid))  # shape (Nx, Nout)
    transformed_values = np.tensordot(sampled_values, phase_matrix, axes=([-1], [0])) * abs(dx)  # integrates along last axis

    return transformed_values, output_grid








