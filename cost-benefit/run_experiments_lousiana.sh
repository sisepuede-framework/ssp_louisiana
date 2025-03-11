#!/bin/bash 
N_EXPERIMENTS=$1
TOTAL_PROCS=$2
PROC_ID=$3

get_array(){
 CHUNK=$((N_EXPERIMENTS / TOTAL_PROCS))
 INICIO=$(( (CHUNK * (PROC_ID-1)) + 1 ))

 echo ""
 if [[  $PROC_ID != $TOTAL_PROCS ]]; then
    FIN=$(( INICIO + (CHUNK) -1))
 else
    FIN=$N_EXPERIMENTS
 fi

 ARREGLO=$(seq $INICIO $FIN)

}

get_array

for i in ${ARREGLO[@]}; do
  echo "Ejecutando experimento $i"
  python cb_lousiana.py $i
done
