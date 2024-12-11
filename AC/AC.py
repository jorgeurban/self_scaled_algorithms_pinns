# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:03:24 2024

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
from scipy.io import loadmat
import json

def load_hparams(file_config):
    with open(file_config, 'r') as file:
        config = json.load(file)
    return config

config = load_hparams('config_AC.json')

seed = config["seed"]["seed"] #Seed

#--------------- Architecture hyperparameters ----------------------------------
neurons = config["architecture_hparams"]["neurons"] #Neurons in every hidden layer
layers = config["architecture_hparams"]["layers"] #Hidden layers
output_dim = config["architecture_hparams"]["output_dim"] #Output dimension
kmax = config["architecture_hparams"]["kmax"] #Maximum wavelength if Fourier inputs are considered
#-------------------------------------------------------------------------------

#--------------- PDE hyperparameters ------------------------------------------
k = config["PDE_hparams"]["k"]
eps = config["PDE_hparams"]["eps"]
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
Nchange=config["batch_hparams"]["Nchange"] #Batch is changed every Nchange iterations
k1 = config["batch_hparams"]["k1"] #k hyperparameter (see adaptive_rad function below)
k2 = config["batch_hparams"]["k2"] #c hyperparameter (see adaptive_rad function below)
x0 = config["batch_hparams"]["x0"] #x0 (minimum value of x)
Lx = config["batch_hparams"]["Lx"] #Lx (length in the x direction)
t0 = config["batch_hparams"]["t0"]
tfinal = config["batch_hparams"]["tfinal"]
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
tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(seed)


class PeriodicC0Layer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PeriodicC0Layer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Assuming inputs is of shape (batch_size, 2) where inputs[:, 0] is t and inputs[:, 1] is x
        t = inputs[:, 0,None]  # Extract t
        x = inputs[:, 1,None]  # Extract x
        ks = tf.convert_to_tensor(np.arange(1,kmax+1),dtype=tf.float64)
        Xper = 2*np.pi*tf.matmul(x,ks[None,:])/Lx
        xcos = tf.math.cos(Xper)
        xsin = tf.math.sin(Xper)
        Xper = tf.concat([xcos,xsin],axis=1)
        Xtot = tf.concat([t,Xper],axis=1)
        return Xtot

def generate_model(layer_dims): 
    '''
    Parameters
    ----------
    layer_dims : TUPLE 
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
    X = PeriodicC0Layer()(X_input)
    X = Dense(layer_dims[1],activation="tanh")(X)
    if len(layer_dims) > 3:
        for i in range(2,len(layer_dims)-1):
            X = Dense(layer_dims[i],activation="tanh")(X)
    X = Dense(layer_dims[-1],activation=None)(X)
    return Model(inputs=X_input,outputs=X)

def generate_inputs(Nint):
    '''
    
    Parameters
    ----------
    Nint : INTEGER
        DESCRIPTION.
        Number of training points in a given batch

    Returns
    -------
    X: Batch of points (in Tensorflow format)
    TYPE : TENSOR
        DESCRIPTION.

    '''
    t = (tfinal-t0)*np.random.rand(Nint) + t0
    x = Lx*np.random.rand(Nint) + x0
    X = np.hstack((t[:,None],x[:,None]))
    return convert_to_tensor(X)

#-----------ADAPTIVE RAD GENERATOR (GIVEN AT WU ET AL., 2023)------------------
def adaptive_rad(N,Nint,rad_args,k,eps,Ntest=100000):
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        DESCRIPTION.
        PINN model, obtained with generate_model() function
        
    Nint : INTEGER
           DESCRIPTION.
           Number of training points in a given batch
        
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
    X: Batch of points (in Tensorflow format)
    TYPE
        DESCRIPTION.
    '''
    Xtest = generate_inputs(Ntest)
    k1,k2 = rad_args
    Y = tf.math.abs(get_results(N,Xtest,k,eps)[-1]).numpy()
    err_eq = np.power(Y,k1)/np.power(Y,k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                         p=err_eq_normalized) 
    return tf.gather(Xtest,X_ids)


#---------------- PINN OUTPUT -------------------------------------------------
def output(N,X):
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
    Returns
    -------
    u: TENSOR
        PINN prediction for u

    '''
    Nout = N(X)
    return Nout[:,0,None]

#-------------------- PDEs ----------------------------------------------------
def get_results(N,X,k,eps):
   '''

   Parameters
   ----------
   N : TENSORFLOW MODEL
       PINN model, obtained with generate_model() function
   X : TENSOR
       Batch of points (in Tensorflow format)
       
   k : TYPE
        Coefficient of the nonlinear part 
   eps : TYPE
        Coefficient that multiplies uxx

   Returns
   -------
   u : TENSOR
       PINN prediction for u
   fu : TENSOR
        PDE residuals for u_t = ...

   '''
   with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
      gt1.watch(X)

      with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
         gt2.watch(X)

         # Calculate u,v
         u = output(N,X)
      
      ugrad = gt2.gradient(u, X)
      u_t = ugrad[:,0]
      u_x = ugrad[:,1]
      
   u_xx = gt1.gradient(u_x, X)[:,1]
   fu = u_t - eps*u_xx +k*(u[:,0]**3-u[:,0])
   return u,fu

#------------------------------- LOSS FUNCTION --------------------------------   
loss_function = keras.losses.MeanSquaredError() #MSE loss
lam0 = 10.
def loss(fu,u0,u0pinn):
    '''
    

    Parameters
    ----------
    fu : TENSOR
         PDE residuals for u_t = ...
    u0 : TENSOR
         Initial condition.
    u0pinn : TENSOR
        PINN prediction of initial condition

    Returns
    -------
    TENSOR
       Total loss function. The term regarding the initial condition is controlled by lam0

    '''
    Ntot = fu.shape[0]
    zeros = tf.zeros([Ntot,1],dtype=tf.float64)
    loss_init = loss_function(u0,u0pinn)
    return loss_function(fu,zeros) + lam0*loss_init

#---------------------- Gradients wrt the trainable parameters ----------------
def grads(N,X,u0,X0,k,eps): 
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
         Batch of points (in Tensorflow format)
    u0 : TENSOR
         Initial condition.
    X0 : TENSOR
        Batch of points at t=0 (in Tensorflow format)
    k : TYPE
         Coefficient of the nonlinear part 
    eps : TYPE
         Coefficient that multiplies uxx

    Returns
    -------
    gradsN : TENSOR
             Gradients of loss wrt trainable variables
    loss_value: FLOAT64
                MSE Loss of PDE residuals

    '''
    with tf.GradientTape() as tape2:
        _,fu = get_results(N,X,k,eps)
        u0pinn = output(N, X0)
        loss_value = loss(fu,u0,u0pinn)
    gradsN = tape2.gradient(loss_value,N.trainable_variables)
    return gradsN,loss_value

#-------------------------TRAINING STEP ---------------------------------------
@tf.function #Precompile training function to accelerate the process
def training(N,X,u0,X0,k,eps,optimizer): #Training step function
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
    u0 : TENSOR
         Initial condition.
    X0 : TENSOR
        Batch of points at t=0 (in Tensorflow format)
    k : TYPE
         Coefficient of the nonlinear part 
    eps : TYPE
         Coefficient that multiplies uxx
        
    Optimizer: TENSORFLOW OPTIMIZER 
               Tensorflow optimizer (Adam, Nadam,...)

    Returns
    -------
    loss_value: FLOAT64
                MSE Loss of PDE residuals
    '''
    parameter_gradients,loss_value = grads(N,X,u0,X0,k,eps)
    optimizer.apply_gradients(zip(parameter_gradients,N.trainable_variables))
    return loss_value

rad_args = (k1,k2) 
epochs = np.arange(Nprint_adam,Adam_epochs+Nprint_adam,Nprint_adam)
loss_list = np.zeros(len(epochs)) #loss list
X = generate_inputs(Nint)    

layer_dims = [None]*(layers + 2)
layer_dims[0] = X.shape[1]
for i in range(1,len(layer_dims)):
    layer_dims[i] = neurons
layer_dims[-1] = output_dim

N = generate_model(layer_dims)

lr = tf.keras.optimizers.schedules.ExponentialDecay(lr0,decay_steps,decay_rate)

optimizer = Adam(lr,b1,b2,epsilon=epsilon) 
template = 'Epoch {}, loss: {}'

data = loadmat("AC_cosine.mat")
unum = data["uu"]
u0 = convert_to_tensor(unum[:,0,None])
X0 = data["x"][0]
T0 = np.zeros(len(X0))
X0 = convert_to_tensor(np.hstack((T0[:,None],X0[:,None])))

#START HERE WITH Adam CONSIDERING A LOWER VALUE OF K, AND THEN MOVE TO THE ORIGINAL PROBLEM.
k=1.
eps=1e-4

#------------------- Adam TRAINING LOOP ---------------------------------------
for i in range(Adam_epochs):
    if (i+1)%Nchange == 0:
        X = adaptive_rad(N,Nint, rad_args,k,eps)
    if (i+1)%Nprint_adam == 0:
        _,fu = get_results(N,X,k,eps)
        u0pinn = output(N, X0)
        loss_value = loss(fu,u0,u0pinn)
        print("i=",i+1)
        print(template.format(i+1,loss_value))
        loss_list[i//Nprint_adam] = loss_value.numpy()
        
    training(N,X,u0,X0,k,eps,optimizer)
    
#RETURN TO THE ORIGINAL PROBLEM
k = config["PDE_hparams"]["k"]
eps = config["PDE_hparams"]["eps"]

np.savetxt("loss_adam_AC.txt",np.c_[epochs,loss_list])
initial_weights = np.concatenate([tf.reshape(w, [-1]).numpy() \
                                  for w in N.weights]) #initial set of trainable variables
layer_dims[0] = 1 + 2*kmax

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

@tf.function
def loss_and_gradient_TF(N,X,u0,X0,k,eps,use_sqrt,use_log): 
    with tf.GradientTape() as tape:
       _,fu = get_results(N,X,k,eps)
       u0pinn = output(N, X0)
       if use_sqrt:
           loss_value = tf.math.sqrt(loss(fu,u0,u0pinn))
       elif use_log:
           loss_value = tf.math.log(loss(fu,u0,u0pinn))
       else:
           loss_value = loss(fu,u0,u0pinn)
    gradsN = tape.gradient(loss_value,N.trainable_variables)
    return loss_value,gradsN

#LOSS AND GRADIENT IN NUMPY FORMAT
def loss_and_gradient(weights,N,X,u0,X0,k,eps,layer_dims,use_sqrt,use_log):
    resh_weights = nested_tensor(weights,layer_dims)
    N.set_weights(resh_weights)
    loss_value,grads = loss_and_gradient_TF(N,X,u0,X0,k,eps,use_sqrt,use_log)
    grads_flat = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grads])
    return loss_value.numpy(), grads_flat

data = loadmat("AC_cosine.mat")
u_ref = data["uu"]
t_star = data["tt"].flatten()
x_star = data["x"].flatten()
x_star,t_star = np.meshgrid(x_star,t_star)

epochs_bfgs = np.arange(0,Nbfgs+100,100) #iterations bfgs list
epochs_bfgs+=Adam_epochs
lossbfgs = np.zeros(len(epochs_bfgs)) #loss bfgs list

X_star = tf.convert_to_tensor(np.hstack((t_star.flatten()[:,None],x_star.flatten()[:,None])))
error_list = np.zeros(len(epochs_bfgs))
cont=0
def callback(*,intermediate_result): 
    global N,cont,lossbfgs,Nprint_bfgs,u_ref,X_star,x_star,\
           error_list
    if (cont+1)%Nprint_bfgs == 0 or cont == 0:
        
        if use_sqrt:
            loss_value = np.power(intermediate_result.fun,2)
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
        lossbfgs[(cont+1)//Nprint_bfgs] = loss_value
        
        utest = output(N,X_star).numpy().reshape(x_star.shape)
        erroru = np.linalg.norm(u_ref.T-utest)/np.linalg.norm(utest)
        error_list[(cont+1)//Nprint_bfgs] = erroru
        print(loss_value,cont+1,erroru)
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
             args = (N,X,u0,X0,k,eps,layer_dims,use_sqrt,use_log),
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
    
    X = adaptive_rad(N,Nint, rad_args,k,eps)
    
if use_sqrt:
    fname_loss = f"AC_loss_{method}_{method_bfgs}_sqrt.txt"
    fname_error = f"AC_error_{method}_{method_bfgs}_sqrt.txt"
elif use_log:
    fname_loss = f"AC_loss_{method}_{method_bfgs}_log.txt"
    fname_error = f"AC_error_{method}_{method_bfgs}_log.txt"
else:
    fname_loss = f"AC_loss_{method}_{method_bfgs}.txt"
    fname_error = f"AC_error_{method}_{method_bfgs}.txt"
    
np.savetxt(fname_loss,np.c_[epochs_bfgs,lossbfgs])
np.savetxt(fname_error,np.c_[epochs_bfgs,error_list])