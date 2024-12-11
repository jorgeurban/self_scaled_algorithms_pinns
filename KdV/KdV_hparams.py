# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:24:04 2024

@author: USUARIO
"""

import json
import numpy as np

def hyperparameter_configuration():
    
    seed={"seed":2}
    architecture_hparams={"neurons":30,
                          "layers":3,
                          "output_dim":1
                          }
    
    PDE_hparams={"c1":6.,
                 "c2":2.,
                 "x1":-2.,
                 "x2":2.
                    }
    
    Adam_hparams={"Adam_epochs":10000,
                  "lr0":1e-2,
                  "decay_steps":1000,
                  "decay_rate":0.98,
                  "b1":0.99,
                  "b2":0.999,
                  "epsilon":1e-20,
                  "Nprint_adam":100
                  }
    
    batch_hparams={"Nint":15000,
                   "Nb":1000,
                  "Nchange":500,
                  "k1":0.,
                  "k2":1.,
                  "x0":0.,
                  "t0":0.,
                  "Lx":20.,
                  "tfinal":5.
                  }
    
    test_hparams={"Nt":600,
                  "Nx":600}
    
    bfgs_hparams={"BFGS_epochs":20000,
                  "method":"BFGS",
                  "method_bfgs":"BFGS",
                  "use_sqrt":False,
                  "use_log":False,
                  "Nprint_bfgs":100}
    
    hparams={
        "seed":seed,
        "architecture_hparams":architecture_hparams,
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "test_hparams":test_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_KdV.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_KdV.json'.")
    
hyperparameter_configuration()