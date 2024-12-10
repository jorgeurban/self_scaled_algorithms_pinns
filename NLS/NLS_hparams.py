# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 21:32:10 2024

@author: USUARIO
"""

import json
import numpy as np

def hyperparameter_configuration():
    
    seed={"seed":2}
    architecture_hparams={"neurons":40,
                          "layers":2,
                          "output_dim":2,
                          "kmax":1
                          }
    
    Adam_hparams={"Adam_epochs":10000,
                  "lr0":5e-3,
                  "decay_steps":1000,
                  "decay_rate":0.98,
                  "b1":0.99,
                  "b2":0.999,
                  "epsilon":1e-20,
                  "Nprint_adam":100
                  }
    
    batch_hparams={"Nint":10000,
                  "Nchange":500,
                  "k1":1.,
                  "k2":1.,
                  "x0":-15.,
                  "t0":0.,
                  "Lx":30.,
                  "tfinal":np.pi/2
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
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_NLS.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_NLS.json'.")
    
hyperparameter_configuration()