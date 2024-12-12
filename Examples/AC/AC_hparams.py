# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 12:30:45 2024

@author: USUARIO
"""

import json
import numpy as np

def hyperparameter_configuration():
    
    seed={"seed":5}
    architecture_hparams={"neurons":30,
                          "layers":3,
                          "output_dim":1,
                          "kmax":1
                          }
    
    PDE_hparams={"k":5.,
                 "eps":1e-4}
    
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
                  "Nchange":500,
                  "k1":1.,
                  "k2":0.,
                  "x0":-1.,
                  "t0":0.,
                  "Lx":2.,
                  "tfinal":1.
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
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_AC.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_AC.json'.")
    
hyperparameter_configuration()