# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:46:56 2024

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

config = load_hparams('config_KdV.json')

seed = config["seed"]["seed"] #Seed

#--------------- Architecture hyperparameters ----------------------------------
neurons = config["architecture_hparams"]["neurons"] #Neurons in every hidden layer
layers = config["architecture_hparams"]["layers"] #Hidden layers
output_dim = config["architecture_hparams"]["output_dim"] #Output dimension
#-------------------------------------------------------------------------------

#--------------- PDE parameters ---------------------------------------
c1 = config["PDE_hparams"]["c1"] #Velocity soliton 1
c2 = config["PDE_hparams"]["c2"] #Velocity soliton 2
x1 = config["PDE_hparams"]["x1"] #Initial position soliton 1
x2 = config["PDE_hparams"]["x2"] #Initial position soliton 2
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
Nb = config["batch_hparams"]["Nb"] #Number of points at boundary
Nchange=config["batch_hparams"]["Nchange"] #Batch is changed every Nchange iterations
k1 = config["batch_hparams"]["k1"] #k hyperparameter (see adaptive_rad function below)
k2 = config["batch_hparams"]["k2"] #c hyperparameter (see adaptive_rad function below)
x0 = config["batch_hparams"]["x0"] #x0 (minimum value of x)
Lx = config["batch_hparams"]["Lx"] #Lx (length in the x direction)
t0 = config["batch_hparams"]["t0"]
tfinal = config["batch_hparams"]["tfinal"]
#------------------------------------------------------------------------------

#------------ Test hyperparameters --------------------------------------
Nx = config["test_hparams"]["Nx"] #Number of grid points for test set in x direction
Nt = config["test_hparams"]["Nt"] #Number of grid points for test set in t direction

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

def generate_inputs(Nint,Nb):
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
    x = (xf-x0)*np.random.rand(Nint) + x0
    X = np.hstack((t[:,None],x[:,None]))
    tb = (tfinal-t0)*np.random.rand(Nb) + t0
    xb = np.ones(Nb)*xf
    Xb = np.hstack((tb[:,None],xb[:,None]))
    return convert_to_tensor(X), convert_to_tensor(Xb)

#-----------ADAPTIVE RAD GENERATOR (GIVEN AT WU ET AL., 2023)------------------
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
    Xtest = generate_inputs(Ntest,1)[0]
    k1,k2 = rad_args
    Y = tf.math.abs(get_results(N,Xtest)[-1]).numpy()
    err_eq = np.power(Y,k1)/np.power(Y,k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                         p=err_eq_normalized) 
    return tf.gather(Xtest,X_ids)


def usol(x,t):
    xi1 = x-c1*t-x1
    xi2 = x-c2*t-x2
    num = 2*(c1-c2)*(c1*tf.math.cosh(np.sqrt(c2)*xi2/2)**2 + c2*tf.math.sinh(np.sqrt(c1)*xi1/2)**2)
    den = ((np.sqrt(c1)-np.sqrt(c2))*tf.math.cosh((xi1*np.sqrt(c1)+xi2*np.sqrt(c2))/2)+\
           (np.sqrt(c1)+np.sqrt(c2))*tf.math.cosh((np.sqrt(c1)*xi1-np.sqrt(c2)*xi2)/2))**2
    return num/den

def uteor(X):
    t = X[:,0,None]
    x = X[:,1,None]
    xi1 = x-c1*t-x1
    xi2 = x-c2*t-x2
    num = 2*(c1-c2)*(c1*tf.math.cosh(np.sqrt(c2)*xi2/2)**2 + c2*tf.math.sinh(np.sqrt(c1)*xi1/2)**2)
    den = ((np.sqrt(c1)-np.sqrt(c2))*tf.math.cosh((xi1*np.sqrt(c1)+xi2*np.sqrt(c2))/2)+\
           (np.sqrt(c1)+np.sqrt(c2))*tf.math.cosh((np.sqrt(c1)*xi1-np.sqrt(c2)*xi2)/2))**2
    return num/den

def uteor_x(X):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        ut = uteor(X)
    gradu = tape.gradient(ut,X)
    ut_x = gradu[:,1]
    return ut_x

def u0(x):
    return usol(x,0)

def g1(t):
    return usol(x0,t)

def g2(t):
    return usol(xf,t)

def r(x,t):
    zero = tf.constant(0.,tf.float64)
    return ((x-x0)*(g2(t)-g2(zero)) + (xf-x)*(g1(t)-g1(zero)))/(xf-x0)

def output(X):
    t = X[:,0,None]
    x = X[:,1,None]
    Xnorm = tf.concat([t/tfinal,x/(xf-x0)],axis=1)
    u = u0(x) + r(x,t) + t*(x-x0)*(x-xf)*N(Xnorm)/(xf-x0)
    return u

def get_results(N,X):
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt1:
        gt1.watch(X)
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt2:
            gt2.watch(X)
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as gt3:
                gt3.watch(X)
                u = output(X)
                
            ugrad = gt3.gradient(u, X)
            u_t = ugrad[:,0]
            u_x = ugrad[:,1]
            
        u_xx = gt2.gradient(u_x, X)[:,1]
        
    u_xxx = gt1.gradient(u_xx,X)[:,1]
   
    fu = u_t + 6*u[:,0]*u_x + u_xxx
    return u,fu

def u_xpinn(X):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X)
        utpinn = output(X)
    gradupinn = tape.gradient(utpinn,X)
    ut_xpinn = gradupinn[:,1]
    return ut_xpinn

loss_function = keras.losses.MeanSquaredError()
def loss(fu,uxf,uxfpinn):
    Ntot = fu.shape[0]
    zeros = tf.zeros([Ntot,1],dtype=tf.float64)
    return loss_function(fu,zeros) + 5*loss_function(uxf,uxfpinn)

def grads(N,X,Xb,uxf): #Gradients wrt the trainable parameters
    with tf.GradientTape() as tape2:
        _,fu = get_results(N,X)
        uxfpinn = u_xpinn(Xb)
        loss_value = loss(fu,uxf,uxfpinn)
    gradsN = tape2.gradient(loss_value,N.trainable_variables)
    return gradsN,loss_value

@tf.function(jit_compile=True) #Precompile training function to accelerate the process
def training(N,X,Xb,uxf,optimizer): #Training step function
    parameter_gradients,loss_value = grads(N,X,Xb,uxf)
    optimizer.apply_gradients(zip(parameter_gradients,N.trainable_variables))
    return loss_value


rad_args = (k1,k2) 
epochs = np.arange(Nprint_adam,Adam_epochs+Nprint_adam,Nprint_adam)
loss_list = np.zeros(len(epochs)) #loss list
X,Xb = generate_inputs(Nint,Nb)    
uxf = uteor_x(Xb)

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
        Xb = generate_inputs(Nint, Nb)[-1]
        uxf = uteor_x(Xb)
        #X = random_permutation(X)
    if (i+1)%Nprint_adam == 0:
        _,fu = get_results(N,X)
        uxfpinn = u_xpinn(Xb)
        loss_value = loss(fu,uxf,uxfpinn)
        print("i=",i+1)
        print(template.format(i+1,loss_value))
        loss_list[i//Nprint_adam] = loss_value.numpy()
        
    training(N,X,Xb,uxf,optimizer)


np.savetxt("loss_adam_NLS.txt",np.c_[epochs,loss_list])
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
def loss_and_gradient_TF(N,X,Xb,uxf,use_sqrt,use_log): #Gradients wrt the trainable parameters
    '''
    Parameters
    ----------
    weights : NUMPY ARRAY
        DESCRIPTION: Trainable parameters in Numpy array format
    N : TENSORFLOW MODEL
        PINN model
    X : TENSOR
        Batch of training params
    Xb: TENSOR
        Batch of boundary inputs
    uxf: TENSOR 
         First derivative at boundary
    use_sqrt : BOOL
        DESCRIPTION. If the square root of the MSE residuals is used for training
    use_log : BOOL
        DESCRIPTION. If the log of the MSE residuals is used for training

    Returns
    -------
    loss_value : LOSS (FLOAT64)
                 LOSS USED FOR TRAINING
    gradsN : ARRAY OF FLOAT64 (TENSORFLOW)
                 Gradients wrt trainable variableS
    '''
    with tf.GradientTape() as tape:
       _,fu = get_results(N,X)
       uxfpinn = u_xpinn(Xb)
       if use_sqrt:
           loss_value = tf.math.sqrt(loss(fu,uxf,uxfpinn))
       elif use_log:
           loss_value = tf.math.log(loss(fu,uxf,uxfpinn))
       else:
           loss_value = loss(fu,uxf,uxfpinn)
    gradsN = tape.gradient(loss_value,N.trainable_variables)
    return loss_value,gradsN

def loss_and_gradient(weights,N,X,Xb,uxf,use_sqrt,use_log,layer_dims):
    '''
    Parameters
    ----------
    weights : NUMPY ARRAY
        DESCRIPTION: Trainable parameters in Numpy array format
    N : TENSORFLOW MODEL
        PINN model
    X : TENSOR
        Batch of training params
    Xb: TENSOR
        Batch of boundary inputs
    uxf: TENSOR 
         First derivative at boundary
    use_sqrt : BOOL
        DESCRIPTION. If the square root of the MSE residuals is used for training
    use_log : BOOL
        DESCRIPTION. If the log of the MSE residuals is used for training
    layer_dims : TUPLE 
        DESCRIPTION: 
        (L0,L1,...,Ln), where Li is the number of neurons at ith layer 
                        if i=0, Li corresponds to the input dimension

    Returns
    -------
    loss_value : LOSS (FLOAT64)
                 LOSS USED FOR TRAINING
    gradsN : ARRAY OF FLOAT64 (TENSORFLOW)
                 Gradients wrt trainable variableS
    '''
    resh_weights = nested_tensor(weights,layer_dims)
    N.set_weights(resh_weights)
    loss_value,grads = loss_and_gradient_TF(N,X,Xb,uxf,use_sqrt,use_log)
    grads_flat = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grads])
    return loss_value.numpy(), grads_flat


def generate_test(Nt,Nx):
    t = np.linspace(t0,tfinal,Nt)
    x = np.linspace(x0,xf,Nx)
    t,x = np.meshgrid(t,x)
    X = np.hstack((t.flatten()[:,None],x.flatten()[:,None]))
    return convert_to_tensor(X),t,x


Xtest,t,x = generate_test(Nt,Nx)

epochs_bfgs = np.arange(0,Nbfgs+Nprint_bfgs,Nprint_bfgs) #iterations bfgs list
epochs_bfgs+=Adam_epochs
lossbfgs = np.zeros(len(epochs_bfgs)) #loss bfgs list
error_list = np.zeros(len(lossbfgs))

cont=0
def callback(*,intermediate_result): #Callback function, to obtain the loss at every iteration
    global N,cont,lossbfgs,Xtest,t,x,error_list,Nprint_bfgs
    if (cont+1)%Nprint_bfgs == 0 or cont == 0:
        if use_sqrt:
            loss_value = np.power(intermediate_result.fun,2)
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
            
        lossbfgs[(cont+1)//Nprint_bfgs] = loss_value
        
        utest = output(Xtest).numpy().reshape(t.shape)
        uteor = usol(x,t)
        error = np.linalg.norm(utest-uteor)/np.linalg.norm(uteor)
        error_list[(cont+1)//Nprint_bfgs] = error
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
    

while cont < Nbfgs: #Training loop
    print(cont)
    result = minimize(loss_and_gradient,initial_weights, 
             args = (N,X,Xb,uxf,use_sqrt,use_log,layer_dims),
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
    Xb = generate_inputs(Nint, Nb)[-1]
    uxf = uteor_x(Xb)

if use_sqrt:
    fname_loss = f"KdV_loss_{method}_{method_bfgs}_sqrt.txt"
    fname_error = f"KdV_error_{method}_{method_bfgs}_sqrt.txt"
elif use_log:
    fname_loss = f"KdV_loss_{method}_{method_bfgs}_log.txt"
    fname_error = f"KdV_error_{method}_{method_bfgs}_log.txt"
else:
    fname_loss = f"KdV_loss_{method}_{method_bfgs}.txt"
    fname_error = f"KdV_error_{method}_{method_bfgs}.txt"
    
np.savetxt(fname_loss,np.c_[epochs_bfgs,lossbfgs])
np.savetxt(fname_error,np.c_[epochs_bfgs,error_list])