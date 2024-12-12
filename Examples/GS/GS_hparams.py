# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:39:58 2024

@author: USUARIO
"""

import json

def hyperparameter_configuration():
    
    seed={"seed":2}
    architecture_hparams={"neurons":30,
                          "layers":2,
                          "output_dim":1,
                          }
    PDE_hparams={"sigma":2.,
                    "s":1.5,
                    "Pc":0.4
                    }
    
    Adam_hparams={"Adam_epochs":5000,
                  "lr0":1e-2,
                  "decay_steps":2000,
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
                  }
    
    bfgs_hparams={"BFGS_epochs":20000,
                  "method":"bfgsz",
                  "method_bfgs":"BFGS",
                  "use_sqrt":False,
                  "use_log":True,
                  "Nprint_bfgs":100}
    
    hparams={
        "seed":seed,
        "architecture_hparams":architecture_hparams,
        "PDE_hparams":PDE_hparams,
        "Adam_hparams":Adam_hparams,
        "batch_hparams":batch_hparams,
        "bfgs_hparams":bfgs_hparams}
    
    # Guarda los hiperparámetros en un archivo JSON
    with open('config.json', 'w') as params:
        json.dump(hparams, params, indent=4)
    
    print("Hiperparámetros guardados en 'config.json'.")
    
hyperparameter_configuration()