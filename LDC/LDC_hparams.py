# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:48:47 2024

@author: USUARIO
"""

import json
import numpy as np

def hyperparameter_configuration():
    
    seed={"seed":1}
    architecture_hparams={"neurons":20,
                          "layers":6,
                          "output_dim":2
                          }
    
    PDE_hparams={"Re":1000.}
    
    Adam_hparams={"Adam_epochs":5000,
                  "lr0":5e-3,
                  "decay_steps":1000,
                  "decay_rate":0.98,
                  "b1":0.99,
                  "b2":0.999,
                  "epsilon":1e-20,
                  "Nprint_adam":100
                  }
    
    batch_hparams={"Nint":15000,
                   "Nbound":5000,
                   "Ncorner":5000,
                   "Nchange":500,
                   "k1":1.,
                   "k2":0.,
                   "x0":0.,
                   "Lx":1.,
                    "y0":0.,
                   "Ly":1.
                  }
    
    bfgs_hparams={"BFGS_epochs":20000,
                  "method":"BFGS",
                  "method_bfgs":"SSBroyden2",
                  "use_sqrt":False,
                  "use_log":False,
                  "Nprint_bfgs":100}
    
    hparams={
        "seed":seed,
        "architecture_hparams":architecture_hparams,
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_LDC.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_LDC.json'.")
    
hyperparameter_configuration()