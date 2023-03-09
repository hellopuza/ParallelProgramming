#!/bin/bash

#PBS -l walltime=00:05:00,nodes=7:ppn=4
#PBS -N test
#PBS -q batch

cd $PBS_O_WORKDIR
mpirun --hostfile $PBS_NODEFILE -np 7 ./number_e 1000000