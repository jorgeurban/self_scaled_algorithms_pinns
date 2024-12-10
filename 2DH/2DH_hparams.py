# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:56:13 2024

@author: USUARIO
"""

import json

def hyperparameter_configuration():
    
    seed={"seed":2}
    architecture_hparams={"neurons":30,
                          "layers":3,
                          "output_dim":1,
                          "kmax":1
                          }
    PDE_hparams={"a1":6.,
                 "a2":6.,
                 "k":1.
                    }
    
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
                  "y0":-1.,
                  "Lx":2.,
                  "Ly":2.
                  }
    
    bfgs_hparams={"BFGS_epochs":50000,
                  "method":"BFGS",
                  "method_bfgs":"SSBroyden2",
                  "use_sqrt":False,
                  "use_log":False,
                  "Nprint_bfgs":100}
    
    test_hparams={"Nx":100,
                  "Ny":100}
    
    hparams={
        "seed":seed,
        "architecture_hparams":architecture_hparams,
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "test_hparams":test_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_2DH.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_2DH.json'.")
    
hyperparameter_configuration()