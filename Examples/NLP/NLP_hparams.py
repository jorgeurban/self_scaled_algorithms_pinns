# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 20:53:00 2024

@author: USUARIO
"""

import json

def hyperparameter_configuration():
    
    seed={"seed":2}
    architecture_hparams={"neurons":30,
                          "layers":2,
                          "output_dim":1
                          }
    PDE_hparams={"k":4.
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
    
    batch_hparams={"Nint":8000,
                  "Nchange":500,
                  "k1":0.,
                  "k2":1.,
                  "x0":0.,
                  "y0":0.,
                  "Lx":1.,
                  "Ly":1.
                  }
    
    bfgs_hparams={"BFGS_epochs":20000,
                  "method":"BFGS",
                  "method_bfgs":"BFGS",
                  "use_sqrt":False,
                  "use_log":True,
                  "Nprint_bfgs":100}
    
    test_hparams={"Nx":1000,
                  "Ny":1000}
    
    hparams={
        "seed":seed,
        "architecture_hparams":architecture_hparams,
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "test_hparams":test_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config_NLP.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config_NLP.json'.")
    
hyperparameter_configuration()