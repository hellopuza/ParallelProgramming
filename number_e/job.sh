#!/bin/bash

#PBS -l walltime=00:01:00,nodes=7:ppn=1
#PBS -N number_e
#PBS -q batch

cd $PBS_O_WORKDIR
mpirun --hostfile $PBS_NODEFILE ./number_e