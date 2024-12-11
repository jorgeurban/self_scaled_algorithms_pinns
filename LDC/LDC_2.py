# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:20:18 2024

@author: USUARIO
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow import convert_to_tensor
from tensorflow.keras.optimizers import Adam
from scipy.optimize import minimize
from scipy.linalg import cholesky,LinAlgError
from scipy import sparse
import json

def load_hparams(file_config):
    with open(file_config, 'r') as file:
        config = json.load(file)
    return config

config = load_hparams('config_LDC.json')

seed = config["seed"]["seed"] #Seed

#--------------- Architecture hyperparameters ----------------------------------
neurons = config["architecture_hparams"]["neurons"] #Neurons in every hidden layer
layers = config["architecture_hparams"]["layers"] #Hidden layers
output_dim = config["architecture_hparams"]["output_dim"] #Output dimension
#-------------------------------------------------------------------------------

#--------------- PDE hyperparameters ------------------------------------------
Re = config["PDE_hparams"]["Re"]
#------------------------------------------------------------------------------

#-------------- Adam hyperparameters -------------------------------------------
Adam_epochs = config["Adam_hparams"]["Adam_epochs"] #Adam epochs
lr0 = config["Adam_hparams"]["lr0"] #Initial learning rate (We consider an exponential lr schedule)
decay_steps = config["Adam_hparams"]["decay_steps"] #Decay steps
decay_rate = config["Adam_hparams"]["decay_rate"] #Decay rate
b1 = config["Adam_hparams"]["b1"] #beta1
b2 = config["Adam_hparams"]["b2"] #beta2
epsilon= config["Adam_hparams"]["epsilon"] #epsilon
Nprint_adam = config["Adam_hparams"]["Nprint_adam"] #Adam results will be printed and save every Nprint_adam iters
#------------------------------------------------------------------------------

#------------ Batch hyperparameters -------------------------------------------
Nint = config["batch_hparams"]["Nint"] #Number of points at batch
Nbound = config["batch_hparams"]["Nbound"] #Number of points at the boundary
Ncorner = config["batch_hparams"]["Ncorner"] #Number of points at the corners of top edge. Included at the interior points batch
Nchange=config["batch_hparams"]["Nchange"] #Batch is changed every Nchange iterations
k1 = config["batch_hparams"]["k1"] #k hyperparameter (see adaptive_rad function below)
k2 = config["batch_hparams"]["k2"] #c hyperparameter (see adaptive_rad function below)
x0 = config["batch_hparams"]["x0"] #x0 (minimum value of x)
Lx = config["batch_hparams"]["Lx"] #Lx (length in the x direction)
y0 = config["batch_hparams"]["y0"] #x0 (minimum value of x)
Ly = config["batch_hparams"]["Ly"] #Lx (length in the x direction)
#------------------------------------------------------------------------------

#------------ Quasi-Newton (QN) hyperparameters -------------------------------
Nbfgs = config["bfgs_hparams"]["BFGS_epochs"] #Number of QN iterations
method = config["bfgs_hparams"]["method"] #Method. See below
method_bfgs = config["bfgs_hparams"]["method_bfgs"] #Quasi-Newton algorithm. See below
use_sqrt = config["bfgs_hparams"]["use_sqrt"] #Use square root of the MSE loss to train
use_log = config["bfgs_hparams"]["use_log"] #Use log of the MSE loss to train
Nprint_bfgs = config["bfgs_hparams"]["Nprint_bfgs"] #QN results will be printed and save every Nprint_adam iters

#In method, you can choose between:
    # -BFGS: Here, we include BFGS and the different self-scaled QN methods.
    #        To distinguish between these algorithms, we use method_bfgs. See below
    # -bfgsr: Personal implementation of the factored BFGS Hessian approximations. 
    #         See https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf for details
    #         Very slow, to be optimized.
    # -bfgsz: Personal implementation of the factored inverse BFGS Hessian approximations.
    #         See https://ccom.ucsd.edu/reports/UCSD-CCoM-22-01.pdf for details
    #         Comparable with BFGS in terms of speed.
    
#If method=BFGS, the variable "method_bfgs" chooses the different QN methods.
#The options for this are (see the modified Scipy optimize script):
    #-BFGS_scipy: The original implementation of BFGS of Scipy
    #-BFGS: Equivalent implementation, but faster (avoid repeated calculations in the BFGS formula)
    #-SSBFGS_AB: The Self-scaled BFGS formula, where the tauk coefficient is calculated with 
    #            Al-Baali's formula (Formula 11 of "Unveiling the optimization process in PINNs")
    #-SSBFGS_OL Same, but tauk is calculated with the original choice of Oren and Luenberger (not recommended)
    #-SSBroyden2: Here we use the tauk and phik expressions defined in the paper
    #             (Formulas 13-23 of "Unveiling the optimization process in PINNs")
    #-SSbroyden1: Another possible choice for these parameters (sometimes better, sometimes worse than SSBroyden1)

#------------------------------------------------------------------------------

xf = x0 + Lx
yf = y0 + Ly
tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(seed)

def generate_model(layer_dims): 
    '''
    

    Parameters
    ----------
    layer_dims : LIST OR TUPLE
        DESCRIPTION.
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer 
                        if i=0, Li corresponds to the input dimension
    Returns
    -------
    Model
    TYPE : TENSORFLOW MODEL
        DESCRIPTION.

    '''
    X_input = Input((layer_dims[0],))
    X = Dense(layer_dims[1],activation="tanh")(X_input)
    if len(layer_dims) > 3:
        for i in range(2,len(layer_dims)-1):
            X = Dense(layer_dims[i],activation="tanh")(X)
    X = Dense(layer_dims[-1],activation=None)(X)
    return Model(inputs=X_input,outputs=X)

def u0(x,C0=50):
    '''
    

    Parameters
    ----------
    x : TENSOR
        x coordinate in [0,1]
    C0 : FLOAT
        COEFFICIENT TO SMOOTH THE BOUNDARY CONDITION AT TOP EDGE. The default is 50.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return 1 - tf.math.cosh(C0*(x-0.5))/np.cosh(0.5*C0)

def generate_inputs(Nint):
    '''
    

    Parameters
    ----------
    Nint : INTEGER
           Number of interior points

    Returns
    -------
    TENSOR
        Interior points

    '''
    y = Lx*np.random.rand(Nint) + y0
    x = Ly*np.random.rand(Nint) + x0
    X = np.hstack((x[:,None],y[:,None]))
    
    return convert_to_tensor(X)

def boundary_inputs(Nb):
    '''
    

    Parameters
    ----------
    Nb : INTEGER
         

    Returns
    -------
    TENSORS
        xy_bnd Points at the boundary
        uv_bnd Values for u and v at the boundary

    '''
    xy_ub = np.random.rand(Nb//2, 2)  # top-bottom boundaries
    xy_ub[:,1] = np.round(xy_ub[:,1])          # y-position is 0 or 1
    xy_lr = np.random.rand(Nb//2, 2)  # left-right boundaries
    xy_lr[:,0] = np.round(xy_lr[:,0])          # x-position is 0 or 1
    xy_bnd = np.random.permutation(np.concatenate([xy_ub, xy_lr]))
    uv_bnd = np.zeros((Nb, 2))
    uv_bnd[:,0] = u0(xy_bnd[:,0])*np.floor(xy_bnd[:,1])
    return convert_to_tensor(xy_bnd),convert_to_tensor(uv_bnd)

def corner_points(Ncorner,alpha=0.99): 
    '''
    Parameters
    ----------
    Ncorner : INT
              Number of points at the corners of the top edge
    alpha : TYPE, optional
            Corners are defined as (x,y) in [0,1-alpha] x [alpha,1] (left corner)
                                   (x,y) in [alpha,1] x [alpha,1]   (right corner)
    Returns
    -------
    TENSOR
        (x,y) points at corners

    '''
    xl = np.random.uniform(low=0,high=(1-alpha),size=Ncorner//2)
    xr = np.random.uniform(low=alpha,high=1,size=Ncorner//2)
    x = np.concatenate((xl,xr))
    yl = np.random.uniform(low=alpha,high=1,size=Ncorner//2)
    yr = np.random.uniform(low=alpha,high=1,size=Ncorner//2)
    y = np.concatenate((yl,yr))
    X = np.hstack((x[:,None],y[:,None]))
    return convert_to_tensor(X)

#-----------ADAPTIVE RAD GENERATOR (GIVEN AT WU ET AL., 2023)------------------
def adaptive_rad(N,Nint,rad_args,Re,Ntest=50000):
    '''

    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    Nint : INTEGER
        NUMBER OF INTERIOR POINTS AT BATCH
    rad_args: TUPLE (k1,k2)
              DESCRIPTION.
              Adaptive resampling of Wu et al. (2023), formula (2)
              k1: k 
              k2: c
              DOI: https://doi.org/10.1016/j.cma.2022.115671
    
    Ntest: INTEGER
           DESCRIPTION.
           Number of test points to do the resampling
        

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    Xtest = generate_inputs(Ntest)
    k1,k2 = rad_args
    fu,fv = get_results(N,Xtest,Re)[-2:]
    Y = tf.math.sqrt(fu**2 + fv**2).numpy()
    err_eq = np.power(Y,k1)/np.power(Y,k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                         p=err_eq_normalized) 
    return tf.gather(Xtest,X_ids)


def output(N,X):
    '''
    

    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS

    Returns
    -------
    u : TENSOR
        VELOCITY U
    v : TENSOR
        VELOCITY V
    p : TENSOR
        PRESSURE P

    '''
    with tf.GradientTape() as tape:
        tape.watch(X)
        Nout = N(X)
        psi = Nout[:,0,None]
    psigrad = tape.gradient(psi,X)
    psi_x = psigrad[:,0,None]
    psi_y = psigrad[:,1,None]
    u = psi_y
    v = -psi_x
    p = Nout[:,1,None] - N(tf.constant([[0.,0.]],dtype=tf.float64))[:,1,None]
    return u,v,p
   
def get_results(N,X,Re):
   '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS
    Re : FLOAT
         REYNOLDS NUMBER

    Returns
    -------
    u : TENSOR
        VELOCITY U
    v : TENSOR
        VELOCITY V
    fu : TENSOR
        RESIDUALS FOR u_t = ...
    fv : TENSOR
        RESIDUALS for v_t = ...

    ''' 
   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
      gt1.watch(X)

      with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
         gt2.watch(X)

         # Calculate u,v
         u,v,p = output(N,X)
      
      ugrad = gt2.gradient(u, X)
      vgrad = gt2.gradient(v, X)
      pgrad = gt2.gradient(p, X)
      
      u_x = ugrad[:,0]
      u_y = ugrad[:,1]
      v_x = vgrad[:,0]
      v_y = vgrad[:,1]
      p_x = pgrad[:,0]
      p_y = pgrad[:,1]
      
   u_xx = gt1.gradient(u_x,X)[:,0]
   u_yy = gt1.gradient(u_y,X)[:,1]
   
   v_xx = gt1.gradient(v_x,X)[:,0]
   v_yy = gt1.gradient(v_y,X)[:,1]
      
   nu=1/Re
   fu = u[:,0]*u_x + v[:,0]*u_y + p_x - nu*(u_xx + u_yy)
   fv = u[:,0]*v_x + v[:,0]*v_y + p_y - nu*(v_xx + v_yy)
   return u,v,fu,fv

loss_function = keras.losses.MeanSquaredError()
lam = 10
def loss(fu,fv,uv,uv_pinn):
    '''
    Parameters
    ----------
    fu : TENSOR
        RESIDUALS FOR u_t = ...
    fv : TENSOR
        RESIDUALS for v_t = ...
    uv : TENSOR
        u and v at boundary
    uv_pinn : TENSOR
        PINN prediction for u and v 

    Returns
    -------
    loss_val : FLOAT
        LOSS VALUE

    '''
    Ntot = fu.shape[0]
    zeros = tf.zeros([Ntot,1],dtype=tf.float64)
    loss_val = loss_function(fu,zeros) + loss_function(fv,zeros) + lam*loss_function(uv,uv_pinn)
    return loss_val

def grads(N,X,Xb,uv,Re): 
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS
    Xb : TENSOR
        BOUNDARY POINTS
    uv : TENSOR
        u and v at boundary
    Re : FLOAT
         REYNOLDS NUMBER

    Returns
    -------
    gradsN : LIST OF TENSORS
        GRADIENT OF LOSS WRT TRAINABLE VARIABLES
    loss_value : FLOAT64
        LOSS VALUE 

    '''
    with tf.GradientTape() as tape2:
        _,_,fu,fv = get_results(N,X,Re)
        uv_pinn = output(N, Xb)[0:2]
        uv_pinn = tf.concat([uv_pinn[0],uv_pinn[1]],axis=1)
        loss_value = loss(fu,fv,uv,uv_pinn)
    gradsN = tape2.gradient(loss_value,N.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return gradsN,loss_value

@tf.function(jit_compile=True) #Precompile training function to accelerate the process
def training(N,X,Xb,uv,Re,optimizer): #Training step function
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS
    Xb : TENSOR
        BOUNDARY POINTS
    uv : TENSOR
        u and v at boundary
    Re : FLOAT
         REYNOLDS NUMBER
    optimizer: TENSORLFOW OPTIMIZER
         ADAM (OR GRADIENT DESCENT BASED) OPTIMIZER
    Returns
    -------
    loss_value : FLOAT64
        LOSS VALUE 
    '''

    parameter_gradients,loss_value = grads(N,X,Xb,uv,Re)
    optimizer.apply_gradients(zip(parameter_gradients,N.trainable_variables))
    return loss_value

rad_args = (k1,k2) #If random uniform, select k1 = 0
epochs = np.arange(Nprint_adam,Adam_epochs+Nprint_adam,Nprint_adam)
loss_list = np.zeros(len(epochs)) #loss list

X = generate_inputs(Nint) #Initial collocation points
Xcorner = corner_points(Ncorner)
X = tf.concat([X,Xcorner],axis=0)
Xb,uv = boundary_inputs(Nbound)
layer_dims = [None]*(layers + 2)
layer_dims[0] = X.shape[1]
for i in range(1,len(layer_dims)):
    layer_dims[i] = neurons
layer_dims[-1] = output_dim

N = generate_model(layer_dims)

lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0,decay_steps,decay_rate)

optimizer = Adam(lr,b1,b2,epsilon=epsilon) 
template = 'Epoch {}, loss: {}'

for i in range(Adam_epochs):
    if (i+1)%Nchange == 0:
        X = adaptive_rad(N, Nint, rad_args, Re)
        Xb,uv = boundary_inputs(Nbound)
        Xcorner = corner_points(Ncorner)
        X = tf.concat([X,Xcorner],axis=0)
        #X = random_permutation(X)
    if (i+1)%Nprint_adam == 0:
        _,_,fu,fv = get_results(N,X,Re)
        uv_pinn = output(N, Xb)[0:2]
        uv_pinn = tf.concat([uv_pinn[0],uv_pinn[1]],axis=1)
        loss_value = loss(fu,fv,uv,uv_pinn)
        print("i=",i+1)
        print(template.format(i+1,loss_value))
        loss_list[i//Nprint_adam] = loss_value.numpy()
        
    training(N,X,Xb,uv,Re,optimizer)
    
np.savetxt("loss_adam_LDC.txt",np.c_[epochs,loss_list])
initial_weights = np.concatenate([tf.reshape(w, [-1]).numpy() \
                                  for w in N.weights]) #initial set of trainable variables
    
def nested_tensor(tparams,layer_dims):
    '''
    
    Parameters
    ----------
    tparams : NUMPY ARRAY
        DESCRIPTION: Trainable parameters in Numpy array format
    layer_dims : TUPLE 
        DESCRIPTION: 
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer 
                        if i=0, Li corresponds to the input dimension

    Returns
    -------
    temp : LIST 
           List of tensors (Trainable variables in Tensorflow format)

    '''
    temp = [None]*(2*len(layer_dims)-2)
    index = 0
    for i in range(len(temp)):
        if i%2==0:
            temp[i] = np.reshape(tparams[index:index+layer_dims[i//2]*\
                    layer_dims[i//2 +1]],(layer_dims[i//2],
                                             layer_dims[i//2 +1]))
            index+=layer_dims[i//2]*layer_dims[i//2 +1]
        else:
            temp[i] = tparams[index:index+layer_dims[i-i//2]]
            index+=layer_dims[i-i//2]
    return temp

    
@tf.function(jit_compile=True)
def loss_and_gradient_TF(N,X,Xb,uv,Re,use_sqrt,use_log): 
    '''
    Parameters
    ----------
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS
    Xb : TENSOR
        BOUNDARY POINTS
    uv : TENSOR
        u and v at boundary
    Re : FLOAT
         REYNOLDS NUMBER
    use_sqrt: BOOL
         If True, the square root of the loss is used
    use_log: BOOL
         If True, the logarithm of the loss is used

    Returns
    -------
    gradsN : LIST OF TENSORS
        GRADIENT OF LOSS WRT TRAINABLE VARIABLES
    loss_value : FLOAT64
        LOSS VALUE 

    '''
    with tf.GradientTape() as tape:
       _,_,fu,fv = get_results(N,X,Re)
       uv_pinn = output(N, Xb)[0:2]
       uv_pinn = tf.concat([uv_pinn[0],uv_pinn[1]],axis=1)
       if use_sqrt:
           loss_value = tf.math.sqrt(loss(fu,fv,uv,uv_pinn))
       elif use_log:
           loss_value = tf.math.log(loss(fu,fv,uv,uv_pinn))
       else:
           loss_value = loss(fu,fv,uv,uv_pinn)
           
    gradsN = tape.gradient(loss_value,N.trainable_variables,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return loss_value,gradsN

def loss_and_gradient(weights,N,X,Xb,uv,Re,use_sqrt,use_log,layer_dims):
    '''
    

    Parameters
    ----------
    weights : NUMPY ARRAY
        TRAINABLE VARIABLES
    N : TENSORFLOW MODEL
        PINN MODEL
    X : TENSOR
        INTERIOR POINTS
    Xb : TENSOR
        BOUNDARY POINTS
    uv : TENSOR
        u and v at boundary
    Re : FLOAT
         REYNOLDS NUMBER
    use_sqrt: BOOL
         If True, the square root of the loss is used
    use_log: BOOL
         If True, the logarithm of the loss is used
    layer_dims : LIST OR TUPLE
        DESCRIPTION.
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer 
                        if i=0, Li corresponds to the input dimension

    Returns
    -------
    loss_value : FLOAT
        LOSS VALUE
    grads_flat : ARRAY
        GRADIENT OF LOSS WRT TRAINABLE VARIABLES

    '''
    resh_weights = nested_tensor(weights,layer_dims)
    N.set_weights(resh_weights)
    loss_value,grads = loss_and_gradient_TF(N,X,Xb,uv,Re,use_sqrt,use_log)
    grads_flat = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grads])
    return loss_value.numpy(), grads_flat

epochs_bfgs = np.arange(0,Nbfgs+Nprint_bfgs,Nprint_bfgs) #iterations bfgs list
epochs_bfgs+=Adam_epochs
lossbfgs = np.zeros(len(epochs_bfgs)) #loss bfgs list
validationbfgs = np.zeros(len(epochs_bfgs))

Nval = 100000
Nbval = 10000
Xval = generate_inputs(Nval)
Xbval,uvval = boundary_inputs(Nbval)

cont=0
def callback(*,intermediate_result): 
    global N,cont,lossbfgs,Nprint_bfgs,Xval,Xbval,uvval
    if (cont+1)%Nprint_bfgs == 0 or cont == 0:
        if use_sqrt:
            loss_value = np.power(intermediate_result.fun,2)
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
        lossbfgs[(cont+1)//Nprint_bfgs] = loss_value
        _,_,fuval,fvval = get_results(N,Xval,Re)
        uv_pinn_val = output(N, Xbval)[0:2]
        uv_pinn_val = tf.concat([uv_pinn_val[0],uv_pinn_val[1]],axis=1)
        validation_value = loss(fuval,fvval,uvval,uv_pinn_val)
        validationbfgs[(cont+1)//Nprint_bfgs] = validation_value
        print(loss_value,validation_value.numpy(),cont+1)
    cont+=1
    
if method == "BFGS":
    method_bfgs = method_bfgs
    initial_scale=False
    H0 = tf.eye(len(initial_weights),dtype=tf.float64)
    H0 = H0.numpy()
    options={'maxiter':Nchange, 'gtol': 0, "hess_inv0":H0,
    "method_bfgs":method_bfgs, "initial_scale":initial_scale}
    
elif method == "bfgsr":
    R0 = sparse.csr_matrix(np.eye(len(initial_weights)))
    options={"maxiter":Nchange,"gtol":0, "r_inv0":R0}
    
elif method == "bfgsz":
    Z0 = tf.eye(len(initial_weights),dtype=tf.float64)
    Z0 = Z0.numpy()
    options={"maxiter":Nchange,"gtol":0, "Z0":Z0}
    
    
#------------------------- BFGS TRAINING --------------------------------------
while cont < Nbfgs: #Training loop
    result = minimize(loss_and_gradient,initial_weights, 
             args = (N,X,Xb,uv,Re,use_sqrt,use_log,layer_dims),
             method=method,jac=True, options=options,
             tol=0,callback=callback) 
    
    initial_weights = result.x
    
    if method=="BFGS":
        H0 = result.hess_inv
        H0 = (H0 + np.transpose(H0))/2
        try:
            cholesky(H0)
        except LinAlgError:
            H0 = tf.eye(len(initial_weights),dtype=tf.float64)
            H0 = H0.numpy()
            
        options={'maxiter':Nchange, 'gtol': 0, "hess_inv0":H0,
        "method_bfgs":method_bfgs, "initial_scale":initial_scale}
            
    elif method=="bfgsr":
        R0 = result.r_inv
        options={"maxiter":Nchange,"gtol":0, "r_inv0":R0}
        
    elif method == "bfgsz":
        Z0 = result.Z
        options={"maxiter":Nchange,"gtol":0, "Z0":Z0}
    
    X = adaptive_rad(N, Nint, rad_args, Re)
    Xb,uv = boundary_inputs(Nbound)
    Xcorner = corner_points(5000)
    X = tf.concat([X,Xcorner],axis=0)
    
if use_sqrt:
    fname_loss = f"LDC_loss_{method}_{method_bfgs}_sqrt.txt"
    fname_error = f"LDC_validation_{method}_{method_bfgs}_sqrt.txt"
elif use_log:
    fname_loss = f"LDC_loss_{method}_{method_bfgs}_log.txt"
    fname_error = f"LDC_validation_{method}_{method_bfgs}_log.txt"
else:
    fname_loss = f"LDC_loss_{method}_{method_bfgs}.txt"
    fname_error = f"LDC_validation_{method}_{method_bfgs}.txt"
    
np.savetxt(fname_loss,np.c_[epochs_bfgs,lossbfgs])
np.savetxt(fname_error,np.c_[epochs_bfgs,validationbfgs])

