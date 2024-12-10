# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:57:10 2023

@author: Usuario
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

config = load_hparams('config.json')

seed = config["seed"]["seed"] #Seed

#--------------- Architecture hyperparameters ----------------------------------
neurons = config["architecture_hparams"]["neurons"] #Neurons in every hidden layer
layers = config["architecture_hparams"]["layers"] #Hidden layers
output_dim = config["architecture_hparams"]["output_dim"] #Output dimension
#-------------------------------------------------------------------------------

#--------------- Source parameters T(P) ---------------------------------------
sigma = config["PDE_hparams"]["sigma"] #Source parameter sigma
s = config["PDE_hparams"]["s"] #Source parameter s
Pc = config["PDE_hparams"]["Pc"] #Source parameter Pc
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

tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(seed)

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
    X = Dense(layer_dims[1],activation="tanh")(X_input)
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
    q = np.random.rand(Nint)
    theta = np.pi*np.random.rand(Nint)
    mu = np.cos(theta)
    X = np.hstack((q[:,None],mu[:,None]))
    return convert_to_tensor(X)

def adaptive_rad(N,Nint,rad_args,Ntest=50000):
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
    Y = tf.math.abs(get_results(N,Xtest)[-1]).numpy()
    err_eq = np.power(Y,k1)/np.power(Y,k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                         p=err_eq_normalized) 
    return tf.gather(Xtest,X_ids)

def P_t(N,X,n=7): 
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
    n: Additional optional power at the function f in hard-enforcement
    P = q^n*(1-mu^2)*Pmu + q(1-q)(1-mu^2)N(q,mu)
       which controls the weight of the boundary condition at surface
    Returns
    -------
    P: TENSOR
        PINN prediction

    '''
    q = X[:,0,None]
    mu = X[:,1,None]
    Pmu = 1+3*mu/2+(15*mu**2 -3)/6 + (140*mu**3 - 60*mu)/32 + \
           (315*mu**4 -210*mu**2 +15)/40 + (1386*mu**5 -1260*mu**3 +210*mu)/96 + \
            (3003*mu**6 -3465*mu**4 + 945*mu**2 -35 )/112 + \
            (51480*mu**7 - 72072*mu**5 +27720*mu**3 - 2520*mu)/1024
    return q*(1-mu**2)*(q**n*Pmu + (1-q)*N(X))


def get_results(N,X):
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)

    Returns
    -------
    P : TENSOR
        PINN prediction
    gs : TENSOR
         PDE residuals

    '''
    q = X[:,0]
    mu = X[:,1]

    with tf.GradientTape(persistent=True, 
                         watch_accessed_variables=False) as gt1:
       gt1.watch(X)

       with tf.GradientTape(persistent=True, 
                            watch_accessed_variables=False) as gt2:
          gt2.watch(X)

          # Calculate P
          P = P_t (N,X)

       # Calculate 1st derivatives
       Pgrad = gt2.gradient(P, X)
       P_q = Pgrad[:,0]
       P_mu = Pgrad[:,1]

    # Calculate 2nd derivatives
    P_qq = gt1.gradient(P_q, X)[:,0]
    P_mumu = gt1.gradient(P_mu, X)[:,1]
    TTp = s**2*sigma*tf.math.abs(P[:,0]-Pc)**(2*sigma-1)*tf.experimental.numpy.heaviside(P[:,0]-Pc,0)
    gs = q*P_qq + 2*P_q + (1-mu**2)*P_mumu/q + TTp/q**3

    #print(P_q[:,None])
    return P,gs

loss_function = keras.losses.MeanSquaredError()
def loss(GS):
    '''
    Parameters
    ----------
    GS : TENSOR
         PDE residuals

    Returns
    -------
    LOSS
    TYPE: FLOAT64
          MSE Loss of PDE residuals

    '''
    Ntot = GS.shape[0]
    zeros = tf.zeros([Ntot,1],dtype=tf.float64)
    return loss_function(GS,zeros)

def grads(N,X): 
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)

    Returns
    -------
    gradsN : TENSOR
             Gradients of loss wrt trainable variables
    P : TENSOR
        PINN prediction
    gs : TENSOR
         PDE residuals
    loss_value: FLOAT64
                MSE Loss of PDE residuals

    '''
    with tf.GradientTape() as tape2:
        P,GS = get_results(N,X)
        loss_value = loss(GS)
    gradsN = tape2.gradient(loss_value,N.trainable_variables)
    return gradsN,P,GS,loss_value

@tf.function(jit_compile=True) #Precompile training function to accelerate the process
def training(N,X,optimizer): #ADAM Training step function
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
        
    Optimizer: TENSORFLOW OPTIMIZER 
               Tensorflow optimizer (Adam, Nadam,...)

    Returns
    -------
    P : TENSOR
        PINN prediction
    GS : TENSOR
         PDE residuals
    loss_value: FLOAT64
                MSE Loss of PDE residuals
    '''
    parameter_gradients,P,GS,loss_value = grads(N,X)
    optimizer.apply_gradients(zip(parameter_gradients,N.trainable_variables))
    return P,GS,loss_value

rad_args = (k1,k2) 
loss_list = np.array([])
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

for i in range(Adam_epochs):
    if (i+1)%Nchange == 0:
        X = adaptive_rad(N, Nint, rad_args)
    if (i+1)%Nprint_adam == 0:
        GS = get_results(N,X)[-1]
        loss_value = loss(GS)
        print("i=",i+1)
        print(template.format(i+1,loss_value))
        loss_list = np.append(loss_list,loss_value.numpy())
        
    training(N,X,optimizer)
    
epochs = np.arange(Nprint_adam,Adam_epochs+Nprint_adam,Nprint_adam)
np.savetxt("loss_adam_GS.txt",np.c_[epochs,loss_list])
initial_weights = np.concatenate([tf.reshape(w, [-1]).numpy() \
                                  for w in N.weights]) 
lossbfgs = np.array([])
        
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

power = 1.
@tf.function(jit_compile=True)
def loss_and_gradient_TF(N,X,use_sqrt,use_log): #Gradients wrt the trainable parameters
    '''
    Parameters
    ----------
    N : TENSORFLOW MODEL
        PINN model, obtained with generate_model() function
    X : TENSOR
        Batch of points (in Tensorflow format)
    use_sqrt: BOOL
        If the square root of the MSE residuals is used for training
    use_log: BOOL
        If the logarithm of the MSE residuals is used for training

    Returns
    -------
    loss_value : FLOAT64
                 LOSS USED FOR TRAINING
    gradsN: TENSOR
            Gradients wrt trainable variables

    '''
    with tf.GradientTape() as tape:
       GS = get_results(N,X)[-1]
       if use_sqrt:
           loss_value = tf.math.sqrt(loss(GS))
       elif use_log:
           loss_value = tf.math.log(loss(GS))
       else:
           loss_value = loss(GS)
    gradsN = tape.gradient(loss_value,N.trainable_variables)
    return loss_value,gradsN

def loss_and_gradient(weights,N,X,layer_dims,use_sqrt,use_log):
    '''
    Parameters
    ----------
    weights : NUMPY ARRAY
        DESCRIPTION: Trainable parameters in Numpy array format
    N : TENSORFLOW MODEL
        PINN model
    X : TENSOR
        Batch of training params
    layer_dims : TUPLE 
        DESCRIPTION: 
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer 
                        if i=0, Li corresponds to the input dimension
    use_sqrt : BOOL
        DESCRIPTION. If the square root of the MSE residuals is used for training
    use_log : BOOL
        DESCRIPTION. If the log of the MSE residuals is used for training

    Returns
    -------
    loss_value : FLOAT64 (NUMPY)
                 LOSS USED FOR TRAINING
    grads_flat : ARRAY OF FLOAT64 (NUMPY)
                 Gradients wrt trainable variableS
    '''
    resh_weights = nested_tensor(weights,layer_dims)
    N.set_weights(resh_weights)
    loss_value,grads = loss_and_gradient_TF(N,X,use_sqrt,use_log)
    grads_flat = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grads])
    return loss_value.numpy(), grads_flat


cont=0
def callback(*,intermediate_result): #Callback function, to obtain the loss at every iteration
    global N,cont,lossbfgs,use_sqrt,use_log,Nprint_bfgs
    if (cont+1)%Nprint_bfgs == 0 or cont == 0:
        if use_sqrt:
            loss_value = np.power(intermediate_result.fun,2)
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
        lossbfgs = np.append(lossbfgs,loss_value)
        print(cont+1,loss_value)
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

while cont < Nbfgs: #Training BFGS loop
    result = minimize(loss_and_gradient,initial_weights, args = (N,X,layer_dims,use_sqrt,use_log),
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
        
    X = adaptive_rad(N, Nint, rad_args)
    
epochs_bfgs = np.arange(0,Nbfgs+Nprint_bfgs,Nprint_bfgs)
epochs_bfgs+=Adam_epochs
if use_sqrt:
    fname = f"GS_loss_{method}_{method_bfgs}_sqrt.txt"
elif use_log:
    fname = f"GS_loss_{method}_{method_bfgs}_log.txt"
else:
    fname = f"GS_loss_{method}_{method_bfgs}.txt"
    
np.savetxt(fname,np.c_[epochs_bfgs,lossbfgs])

