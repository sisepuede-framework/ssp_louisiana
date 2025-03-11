#!/bin/bash

mkdir cb_resultados/{cb,qa}

bash run_experiments_lousiana.sh 1000 1 1

python cb_resultados/reune_resultados.py
