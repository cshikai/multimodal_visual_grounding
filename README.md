Visual Grounding Model Training
==============================
Project Organization
------------

    ├── README.md               <- The top-level README for developers using this project.
    |
    ├── docker                  <- Folder that contains files for building the docker environment 
    │   ├── Dockerfile          <- Dockerfile for building docker image
    │   ├── Makefile            <- Makefile with commands like `make data` or `make train`. Ran under Dockerfile
    │   └── requirements.txt    <- The requirements file for reproducing the analysis environment, e.g.
    │                           generated with `pip freeze > requirements.txt`
    │
    ├── data        
    │   └── example.parquet     <- Download data from clearml here
    |
    ├── src                     <- Source code for use in this project.
    │   │
    │   ├── main.py             <- Code to run for task initialization,  sending to remote, download datasets, starting experimentation
    |   |
    │   ├── experiment.py       <- Experimentation defining the datasets, trainer, epoch behaviour and running training
    |   |
    │   ├── config.yaml         <- Config file
    |   |
    │   ├── data                <- Scripts related to data procesing
    │   │   ├── dataset.py
    │   │   ├── postprocessing.py
    │   │   ├── preprocessing.py
    |   |   └── common
    |   │       └── transforms.py
    │   │
    |   ├── models              <- Scripts related to module architecture
    |   │   ├── model.py        <- Main model file chaining together modules 
    |   │   ├── config.py       <- Boilerplate code for config loading
    |   │   ├── config.yaml     <- Configfile for model parameters
    |   │   └── modules         <- Folder containing model modules
    |   |       └── common
    |   |           └── crf.py 
    |   |       ├── encoder.py           
    |   |       └── decoder.py           
    │   │
    │   └── evaluation          <- Scripts to generate evaluations of model e.g. confusion matrix etc.
    |       ├── visualize.py           
    |       └── common 
    |           └── metrics.py
    |
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    |
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    |
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

