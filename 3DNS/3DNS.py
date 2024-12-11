# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:49:43 2024

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

config = load_hparams('config_3DNS.json')

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
Nchange=config["batch_hparams"]["Nchange"] #Batch is changed every Nchange iterations
k1 = config["batch_hparams"]["k1"] #k hyperparameter (see adaptive_rad function below)
k2 = config["batch_hparams"]["k2"] #c hyperparameter (see adaptive_rad function below)
x0 = config["batch_hparams"]["x0"] #x0 (minimum value of x)
Lx = config["batch_hparams"]["Lx"] #Lx (length in the x direction)
y0 = config["batch_hparams"]["y0"] #x0 (minimum value of x)
Ly = config["batch_hparams"]["Ly"] #Lx (length in the x direction)
z0 = config["batch_hparams"]["z0"] #x0 (minimum value of x)
Lz = config["batch_hparams"]["Lz"] #Lx (length in the x direction)
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
yf = y0 + Ly
zf = z0 + Lz
tf.keras.backend.set_floatx('float64')
tf.get_logger().setLevel('ERROR')
tf.keras.utils.set_random_seed(seed)

def generate_model(layer_dims): 
    X_input = Input((layer_dims[0],))
    X = Dense(layer_dims[1],activation="tanh")(X_input)
    if len(layer_dims) > 3:
        for i in range(2,len(layer_dims)-1):
            X = Dense(layer_dims[i],activation="tanh")(X)
    X = Dense(layer_dims[-1],activation=None)(X)
    return Model(inputs=X_input,outputs=X)

def generate_inputs(Nint,x0,y0,z0,Lx,Ly,Lz):
    t = np.random.rand(Nint)
    x = Lx*np.random.rand(Nint) + x0
    y = Ly*np.random.rand(Nint) + y0
    z = Lz*np.random.rand(Nint) + z0
    Xcoord = np.hstack((x[:,None],np.hstack((y[:,None],z[:,None]))))
    X = np.hstack((t[:,None],Xcoord))
    return convert_to_tensor(X)

def adaptive_rad(N,Nint,rad_args,x0,y0,z0,Lx,Ly,Lz,Ntest=50000):
    Xtest = generate_inputs(Ntest,x0,y0,z0,Lx,Ly,Lz)
    k1,k2 = rad_args
    Y = tf.math.abs(PDEs(N,Xtest,x0,y0,z0,Lx,Ly,Lz)[-1]).numpy()
    err_eq = np.power(Y,k1)/np.power(Y,k1).mean() + k2
    err_eq_normalized = (err_eq / sum(err_eq))
    X_ids = np.random.choice(a=len(Xtest), size=Nint, replace=False,
                         p=err_eq_normalized) 
    return tf.gather(Xtest,X_ids)

def velocityt(x,y,z,t):
    sinu,cosu = tf.math.sin(y+z),tf.math.cos(x+y)
    sinv,cosv = tf.math.sin(x+z),tf.math.cos(z+y)
    sinw,cosw = tf.math.sin(y+x),tf.math.cos(x+z)
    u = -tf.math.exp(x-t)*sinu - cosu*tf.math.exp(z-t)
    v = -tf.math.exp(y-t)*sinv - cosv*tf.math.exp(x-t)
    w = -tf.math.exp(z-t)*sinw - cosw*tf.math.exp(y-t)
    return u,v,w

def pressure(X):
    t = X[:,0,None]
    x = X[:,1,None]
    y = X[:,2,None]
    z = X[:,3,None]
    sinu,cosu = tf.math.sin(y+z),tf.math.cos(x+y)
    sinv,cosv = tf.math.sin(x+z),tf.math.cos(z+y)
    sinw,cosw = tf.math.sin(y+x),tf.math.cos(x+z)
    return -tf.math.exp(-2*t)*(tf.math.exp(2*x)+tf.math.exp(2*y) + tf.math.exp(2*z) + 2*sinw*cosw*\
             tf.math.exp(y+z) +2*sinu*cosu*tf.math.exp(x+z) + 2*sinv*cosv*tf.math.exp(x+y))/2
     
def boundary_conditions(t,x,y,z,x0,y0,z0,Lx,Ly,Lz):
    
    xi_x = (x-x0)/Lx
    xi_y = (y-y0)/Lx
    xi_z = (z-z0)/Lz
    
    zero = tf.constant(0.,dtype=tf.float64)
    
    ux0,vx0,wx0 = velocityt(x0,y,z,t)
    u0x0,v0x0,w0x0 = velocityt(x0,y,z,zero)
    ux0pl,vx0pl,wx0pl = velocityt(x0+Lx,y,z,t)
    u0x0pl,v0x0pl,w0x0pl = velocityt(x0+Lx,y,z,zero)
    
    uy0,vy0,wy0 = velocityt(x,y0,z,t)
    u0y0,v0y0,w0y0 = velocityt(x,y0,z,zero)
    uy0pl,vy0pl,wy0pl = velocityt(x,y0+Ly,z,t)
    u0y0pl,v0y0pl,w0y0pl = velocityt(x,y0+Ly,z,zero)
    
    uz0,vz0,wz0 = velocityt(x,y,z0,t)
    u0z0,v0z0,w0z0 = velocityt(x,y,z0,zero)
    uz0pl,vz0pl,wz0pl = velocityt(x,y,z0+Lz,t)
    u0z0pl,v0z0pl,w0z0pl = velocityt(x,y,z0+Lz,zero)
    
    fou,fov,fow = (ux0-u0x0,vx0-v0x0,wx0-w0x0)
    f1u,f1v,f1w = (ux0pl-u0x0pl,vx0pl-v0x0pl,wx0pl-w0x0pl)
    
    gou,gov,gow = (uy0-u0y0,vy0-v0y0,wy0-w0y0)
    g1u,g1v,g1w = (uy0pl-u0y0pl,vy0pl-v0y0pl,wy0pl-w0y0pl)
    
    hou,hov,how = (uz0-u0z0,vz0-v0z0,wz0-w0z0)
    h1u,h1v,h1w = (uz0pl-u0z0pl,vz0pl-v0z0pl,wz0pl-w0z0pl)
    
    
    ux0y0,vx0y0,wx0y0 = velocityt(x0,y0,z,t)
    ux0y0pl,vx0y0pl,wx0y0pl = velocityt(x0,y0+Ly,z,t)
    ux0ply0,vx0ply0,wx0ply0 = velocityt(x0+Lx,y0,z,t)
    ux0ply0pl,vx0ply0pl,wx0ply0pl = velocityt(x0+Lx,y0+Ly,z,t)
    
    u0x0y0,v0x0y0,w0x0y0 = velocityt(x0,y0,z,zero)
    u0x0y0pl,v0x0y0pl,w0x0y0pl = velocityt(x0,y0+Ly,z,zero)
    u0x0ply0,v0x0ply0,w0x0ply0 = velocityt(x0+Lx,y0,z,zero)
    u0x0ply0pl,v0x0ply0pl,w0x0ply0pl = velocityt(x0+Lx,y0+Ly,z,zero)
    
    G0u = gou - (1-xi_x)*(ux0y0-u0x0y0) - xi_x*(ux0ply0-u0x0ply0)
    G1u = g1u - (1-xi_x)*(ux0y0pl-u0x0y0pl) - xi_x*(ux0ply0pl-u0x0ply0pl)
    
    G0v = gov - (1-xi_x)*(vx0y0-v0x0y0) - xi_x*(vx0ply0-v0x0ply0)
    G1v = g1v - (1-xi_x)*(vx0y0pl-v0x0y0pl) - xi_x*(vx0ply0pl-v0x0ply0pl)
    
    G0w = gow - (1-xi_x)*(wx0y0-w0x0y0) - xi_x*(wx0ply0-w0x0ply0)
    G1w = g1w - (1-xi_x)*(wx0y0pl-w0x0y0pl) - xi_x*(wx0ply0pl-w0x0ply0pl)
    
    ux0z0,vx0z0,wx0z0 = velocityt(x0,y,z0,t)
    ux0z0pl,vx0z0pl,wx0z0pl = velocityt(x0,y,z0+Lz,t)
    u0x0z0,v0x0z0,w0x0z0 = velocityt(x0,y,z0,0.)
    u0x0z0pl,v0x0z0pl,w0x0z0pl = velocityt(x0,y,z0+Lz,0.)
    
    ux0plz0,vx0plz0,wx0plz0 = velocityt(x0+Lx,y,z0,t)
    ux0plz0pl,vx0plz0pl,wx0plz0pl = velocityt(x0+Lx,y,z0+Lz,t)
    u0x0plz0,v0x0plz0,w0x0plz0 = velocityt(x0+Lx,y,z0,0.)
    u0x0plz0pl,v0x0plz0pl,w0x0plz0pl = velocityt(x0+Lx,y,z0+Lz,0.)
    
    uy0z0,vy0z0,wy0z0 = velocityt(x,y0,z0,t)
    uy0z0pl,vy0z0pl,wy0z0pl = velocityt(x,y0,z0+Lz,t)
    u0y0z0,v0y0z0,w0y0z0 = velocityt(x,y0,z0,0.)
    u0y0z0pl,v0y0z0pl,w0y0z0pl = velocityt(x,y0,z0+Lz,0.)
    
    ux0y0z0,vx0y0z0,wx0y0z0 = velocityt(x0,y0,z0,t)
    ux0y0z0pl,vx0y0z0pl,wx0y0z0pl = velocityt(x0,y0,z0+Lz,t)
    u0x0y0z0,v0x0y0z0,w0x0y0z0 = velocityt(x0,y0,z0,0.)
    u0x0y0z0pl,v0x0y0z0pl,w0x0y0z0pl = velocityt(x0,y0,z0+Lz,0.)
    
    ux0ply0z0,vx0ply0z0,wx0ply0z0 = velocityt(x0+Lx,y0,z0,t)
    ux0ply0z0pl,vx0ply0z0pl,wx0ply0z0pl = velocityt(x0+Lx,y0,z0+Lz,t)
    u0x0ply0z0,v0x0ply0z0,w0x0ply0z0 = velocityt(x0+Lx,y0,z0,0.)
    u0x0ply0z0pl,v0x0ply0z0pl,w0x0ply0z0pl = velocityt(x0+Lx,y0,z0+Lz,0.)
    
    uy0plz0,vy0plz0,wy0plz0 = velocityt(x,y0+Ly,z0,t)
    uy0plz0pl,vy0plz0pl,wy0plz0pl = velocityt(x,y0+Ly,z0+Lz,t)
    u0y0plz0,v0y0plz0,w0y0plz0 = velocityt(x,y0+Ly,z0,0.)
    u0y0plz0pl,v0y0plz0pl,w0y0plz0pl = velocityt(x,y0+Ly,z0+Lz,0.)
    
    ux0y0plz0,vx0y0plz0,wx0y0plz0 = velocityt(x0,y0+Ly,z0,t)
    ux0y0plz0pl,vx0y0plz0pl,wx0y0plz0pl = velocityt(x0,y0+Ly,z0+Lz,t)
    u0x0y0plz0,v0x0y0plz0,w0x0y0plz0 = velocityt(x0,y0+Ly,z0,0.)
    u0x0y0plz0pl,v0x0y0plz0pl,w0x0y0plz0pl = velocityt(x0,y0+Ly,z0+Lz,0.)
    
    ux0ply0plz0,vx0ply0plz0,wx0ply0plz0 = velocityt(x0+Lx,y0+Ly,z0,t)
    ux0ply0plz0pl,vx0ply0plz0pl,wx0ply0plz0pl = velocityt(x0+Lx,y0+Ly,z0+Lz,t)
    u0x0ply0plz0,v0x0ply0plz0,w0x0ply0plz0 = velocityt(x0+Lx,y0+Ly,z0,0.)
    u0x0ply0plz0pl,v0x0ply0plz0pl,w0x0ply0plz0pl = velocityt(x0+Lx,y0+Ly,z0+Lz,0.)
    
    H0u = hou - (1-xi_x)*(ux0z0-u0x0z0) - xi_x*(ux0plz0-u0x0plz0) - (1-xi_y)*\
        (uy0z0-u0y0z0 - (1-xi_x)*(ux0y0z0-u0x0y0z0) - xi_x*(ux0ply0z0-u0x0ply0z0))-\
        xi_y*(uy0plz0-u0y0plz0 - (1-xi_x)*(ux0y0plz0-u0x0y0plz0)-xi_x*\
        (ux0ply0plz0-u0x0ply0plz0))
    
    H1u = h1u - (1-xi_x)*(ux0z0pl-u0x0z0pl) - xi_x*(ux0plz0pl-u0x0plz0pl) - (1-xi_y)*\
        (uy0z0pl-u0y0z0pl - (1-xi_x)*(ux0y0z0pl-u0x0y0z0pl) - xi_x*(ux0ply0z0pl-u0x0ply0z0pl))-\
        xi_y*(uy0plz0pl-u0y0plz0pl - (1-xi_x)*(ux0y0plz0pl-u0x0y0plz0pl)-xi_x*\
        (ux0ply0plz0pl-u0x0ply0plz0pl))
            
    H0v = hov - (1-xi_x)*(vx0z0-v0x0z0) - xi_x*(vx0plz0-v0x0plz0) - (1-xi_y)*\
        (vy0z0-v0y0z0 - (1-xi_x)*(vx0y0z0-v0x0y0z0) - xi_x*(vx0ply0z0-v0x0ply0z0))-\
        xi_y*(vy0plz0-v0y0plz0 - (1-xi_x)*(vx0y0plz0-v0x0y0plz0)-xi_x*\
        (vx0ply0plz0-v0x0ply0plz0))
    
    H1v = h1v - (1-xi_x)*(vx0z0pl-v0x0z0pl) - xi_x*(vx0plz0pl-v0x0plz0pl) - (1-xi_y)*\
        (vy0z0pl-v0y0z0pl - (1-xi_x)*(vx0y0z0pl-v0x0y0z0pl) - xi_x*(vx0ply0z0pl-v0x0ply0z0pl))-\
        xi_y*(vy0plz0pl-v0y0plz0pl - (1-xi_x)*(vx0y0plz0pl-v0x0y0plz0pl)-xi_x*\
        (vx0ply0plz0pl-v0x0ply0plz0pl))
            
    H0w = how - (1-xi_x)*(wx0z0-w0x0z0) - xi_x*(wx0plz0-w0x0plz0) - (1-xi_y)*\
        (wy0z0-w0y0z0 - (1-xi_x)*(wx0y0z0-w0x0y0z0) - xi_x*(wx0ply0z0-w0x0ply0z0))-\
        xi_y*(wy0plz0-w0y0plz0 - (1-xi_x)*(wx0y0plz0-w0x0y0plz0)-xi_x*\
        (wx0ply0plz0-w0x0ply0plz0))
    
    H1w = h1w - (1-xi_x)*(wx0z0pl-w0x0z0pl) - xi_x*(wx0plz0pl-w0x0plz0pl) - (1-xi_y)*\
        (wy0z0pl-w0y0z0pl - (1-xi_x)*(wx0y0z0pl-w0x0y0z0pl) - xi_x*(wx0ply0z0pl-w0x0ply0z0pl))-\
        xi_y*(wy0plz0pl-w0y0plz0pl - (1-xi_x)*(wx0y0plz0pl-w0x0y0plz0pl)-xi_x*\
        (wx0ply0plz0pl-w0x0ply0plz0pl))
            
    Au = (1-xi_x)*fou + xi_x*f1u + (1-xi_y)*G0u + xi_y*G1u + (1-xi_z)*H0u + xi_z*H1u
    Av = (1-xi_x)*fov + xi_x*f1v + (1-xi_y)*G0v + xi_y*G1v + (1-xi_z)*H0v + xi_z*H1v
    Aw = (1-xi_x)*fow + xi_x*f1w + (1-xi_y)*G0w + xi_y*G1w + (1-xi_z)*H0w + xi_z*H1w
    return Au,Av,Aw
    
      
def output(N,X,x0,y0,z0,Lx,Ly,Lz):
    t = X[:,0,None]
    x = X[:,1,None]
    y = X[:,2,None]
    z = X[:,3,None]
    Au,Av,Aw = boundary_conditions(t,x,y,z,x0,y0,z0,Lx,Ly,Lz)
    zero = tf.constant(0.,dtype=tf.float64)
    u0,v0,w0 = velocityt(x, y, z, zero)
    Nout = N(X)
    h = t*(x-x0)*(x-x0-Lx)*(y-y0)*(y-y0-Ly)*(z-z0)*(z-z0-Lz)
    
    X0 = tf.concat([tf.zeros([len(t),1],dtype=tf.float64),
    tf.concat([tf.zeros([len(t),1],dtype=tf.float64),
               tf.zeros([len(t),1],dtype=tf.float64)],axis=1)],axis=1)
    X0 = tf.concat([t,X0],axis=1)
    P0 = pressure(X0)
    
    u = u0 + Au + h*Nout[:,0,None]
    v = v0 + Av + h*Nout[:,1,None]
    w = w0 + Aw + h*Nout[:,2,None]
    p = Nout[:,3,None] - N(X0)[:,3,None] + P0
    return u,v,w,p

def PDEs(N,X,x0,y0,z0,Lx,Ly,Lz):
    
    with tf.GradientTape(persistent=True, 
                         watch_accessed_variables=False) as gt1:
       gt1.watch(X)

       with tf.GradientTape(persistent=True, 
                            watch_accessed_variables=False) as gt2:
          gt2.watch(X)
          
          u,v,w,p = output(N,X,x0,y0,z0,Lx,Ly,Lz)
          
       ugrad = gt2.gradient(u,X)
       vgrad = gt2.gradient(v,X)
       wgrad = gt2.gradient(w,X)
       pgrad = gt2.gradient(p,X)
       
       u_t = ugrad[:,0]
       v_t = vgrad[:,0]
       w_t = wgrad[:,0]
       
       u_x = ugrad[:,1]
       v_x = vgrad[:,1]
       w_x = wgrad[:,1]
       
       u_y = ugrad[:,2]
       v_y = vgrad[:,2]
       w_y = wgrad[:,2]
       
       u_z = ugrad[:,3]
       v_z = vgrad[:,3]
       w_z = wgrad[:,3]
       
       p_x = pgrad[:,1]
       p_y = pgrad[:,2]
       p_z = pgrad[:,3]
       
    u_xx = gt1.gradient(u_x,X)[:,1]
    u_yy = gt1.gradient(u_y,X)[:,2]
    u_zz = gt1.gradient(u_z,X)[:,3]
    
    v_xx = gt1.gradient(v_x,X)[:,1]
    v_yy = gt1.gradient(v_y,X)[:,2]
    v_zz = gt1.gradient(v_z,X)[:,3]
    
    w_xx = gt1.gradient(w_x,X)[:,1]
    w_yy = gt1.gradient(w_y,X)[:,2]
    w_zz = gt1.gradient(w_z,X)[:,3]
    
    equ = u_t + u[:,0]*u_x + v[:,0]*u_y + w[:,0]*u_z - (1/Re)*(u_xx + u_yy + u_zz) + p_x
    eqv = v_t + u[:,0]*v_x + v[:,0]*v_y + w[:,0]*v_z - (1/Re)*(v_xx + v_yy + v_zz) + p_y
    eqw = w_t + u[:,0]*w_x + v[:,0]*w_y + w[:,0]*w_z - (1/Re)*(w_xx + w_yy + w_zz) + p_z
    eq_inc = u_x + v_y + w_z 
    
    return equ,eqv,eqw,eq_inc

loss_function = keras.losses.MeanSquaredError()
def loss(equ,eqv,eqw,eq_inc):
    zeros = tf.zeros([len(equ),1],dtype=tf.float64)
    return loss_function(equ,zeros) + loss_function(eqv,zeros) + loss_function(eqw,zeros) + \
        loss_function(eq_inc,zeros)
        
def grads(N,X,x0,y0,z0,Lx,Ly,Lz): #Gradients wrt the trainable parameters
    with tf.GradientTape() as tape2:
        equ,eqv,eqw,eq_inc = PDEs(N,X,x0,y0,z0,Lx,Ly,Lz)
        loss_value = loss(equ,eqv,eqw,eq_inc)
    gradsN = tape2.gradient(loss_value,N.trainable_variables)
    return gradsN,loss_value

@tf.function(jit_compile=True)
def training(N,X,x0,y0,z0,Lx,Ly,Lz,optimizer): #Training step function
    parameter_gradients,loss_value = grads(N,X,x0,y0,z0,Lx,Ly,Lz)
    optimizer.apply_gradients(zip(parameter_gradients,N.trainable_variables))
    return loss_value

x0 = tf.constant(x0,dtype=tf.float64)
y0 = tf.constant(y0,dtype=tf.float64)
z0 = tf.constant(z0,dtype=tf.float64)
Lx = tf.constant(Lx,dtype=tf.float64)
Ly = tf.constant(Ly,dtype=tf.float64)
Lz = tf.constant(Lz,dtype=tf.float64)

rad_args = (k1,k2) #If random uniform, select k1 = 0
epochs = np.arange(Nprint_adam,Adam_epochs+Nprint_adam,Nprint_adam)
loss_list = np.zeros(len(epochs)) #loss list 
X = generate_inputs(Nint,x0,y0,z0,Lx,Ly,Lz)
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
        X = adaptive_rad(N, Nint, rad_args,x0,y0,z0,Lx,Ly,Lz)
        #X = random_permutation(X)
    if (i+1)%Nprint_adam == 0:
        equ,eqv,eqw,eq_inc = PDEs(N,X,x0,y0,z0,Lx,Ly,Lz)
        loss_value = loss(equ,eqv,eqw,eq_inc)
        print("i=",i+1)
        print(template.format(i+1,loss_value))
        loss_list[i//Nprint_adam] = loss_value.numpy()
        
    training(N,X,x0,y0,z0,Lx,Ly,Lz,optimizer)

np.savetxt("loss_adam_3DNS.txt",np.c_[epochs,loss_list])
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
def loss_and_gradient_TF(N,X,use_sqrt,use_log): #Gradients wrt the trainable parameters
    with tf.GradientTape() as tape:
      equ,eqv,eqw,eq_inc = PDEs(N,X,x0,y0,z0,Lx,Ly,Lz)
      if use_sqrt:
          loss_value = tf.math.sqrt(loss(equ,eqv,eqw,eq_inc))
      elif use_log:
          loss_value = tf.math.log(loss(equ,eqv,eqw,eq_inc))
      else:
          loss_value = loss(loss(equ,eqv,eqw,eq_inc))
    gradsN = tape.gradient(loss_value,N.trainable_variables)
    return loss_value,gradsN

def loss_and_gradient(weights,N,X,layer_dims,use_sqrt,use_log):
    resh_weights = nested_tensor(weights,layer_dims)
    N.set_weights(resh_weights)
    loss_value,grads = loss_and_gradient_TF(N,X,use_sqrt,use_log)
    grads_flat = np.concatenate([tf.reshape(g, [-1]).numpy() for g in grads])
    return loss_value.numpy(), grads_flat

def generate_slice(Nx,Ny,zval=0.5,tval=1.):
    x = np.linspace(x0,xf,Nx)
    y = np.linspace(y0,yf,Ny)
    y,x = np.meshgrid(y,x)
    z = zval*np.ones(Nx*Ny)
    t = tval*np.ones(Nx*Ny)
    Xcoord = np.hstack((x.flatten()[:,None],np.hstack((y.flatten()[:,None],z[:,None]))))
    X = np.hstack((t[:,None],Xcoord))
    return convert_to_tensor(X),x,y

Nx = 300
Ny = 300
Xstar,x,y = generate_slice(Nx, Ny)
ustar,vstar,wstar = velocityt(Xstar[:,1], Xstar[:,2], Xstar[:,3], Xstar[:,0])
ustar = ustar.numpy().reshape(x.shape)
vstar = vstar.numpy().reshape(x.shape)
wstar = wstar.numpy().reshape(x.shape)
umod_star = np.sqrt(ustar**2 + vstar**2 + wstar**2)

epochs_bfgs = np.arange(0,Nbfgs+Nprint_bfgs,Nprint_bfgs) #iterations bfgs list
epochs_bfgs+=Adam_epochs
lossbfgs = np.zeros(len(epochs_bfgs)) #loss bfgs list
error_list = np.zeros(len(lossbfgs))

cont=0
def callback(*,intermediate_result): #Callback function, to obtain the loss at every iteration
    global N,cont,lossbfgs,Xstar,umod_star,error_list
    '''
    if cont%Nsave == 0:
        N.save(fname.format(cont+Nepochs)) 
    '''
    if (cont+1)%100 == 0 or cont == 0:
        if use_sqrt:
            loss_value = np.power(intermediate_result.fun,2)
        elif use_log:
            loss_value = np.exp(intermediate_result.fun)
        else:
            loss_value = intermediate_result.fun
        lossbfgs[(cont+1)//Nprint_bfgs] = loss_value
        
        utest,vtest,wtest = output(Xstar, x0, y0, z0, Lx, Ly, Lz)[:-1]
        utest = utest.numpy().reshape(umod_star.shape)
        vtest = vtest.numpy().reshape(umod_star.shape)
        wtest = wtest.numpy().reshape(umod_star.shape)
        umod_test = np.sqrt(utest**2 + vtest**2 + wtest**2)
        error = np.linalg.norm(umod_star-umod_test)/np.linalg.norm(umod_star)
        error_list = np.append(error_list,error)
        #Bk = intermediate_result.hess_inv
        #maxlamb = np.append(maxlamb,np.max(np.linalg.eig(Bk)[0]))
        #minlamb = np.append(minlamb,np.min(np.linalg.eig(Bk)[0]))
        print(cont+1,error,loss_value)
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
    
    X = adaptive_rad(N, Nint, rad_args,x0,y0,z0,Lx,Ly,Lz)
    
if use_sqrt:
    fname_loss = f"3DNS_loss_{method}_{method_bfgs}_sqrt.txt"
    fname_error = f"3DNS_error_{method}_{method_bfgs}_sqrt.txt"
elif use_log:
    fname_loss = f"3DNS_loss_{method}_{method_bfgs}_log.txt"
    fname_error = f"3DNS_error_{method}_{method_bfgs}_log.txt"
else:
    fname_loss = f"3DNS_loss_{method}_{method_bfgs}.txt"
    fname_error = f"3DNS_error_{method}_{method_bfgs}.txt"
    
np.savetxt(fname_loss,np.c_[epochs_bfgs,lossbfgs])
np.savetxt(fname_error,np.c_[epochs_bfgs,error_list])
    