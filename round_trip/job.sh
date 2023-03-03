#!/bin/bash

#PBS -l walltime=00:01:00,nodes=7:ppn=1
#PBS -N round_trip
#PBS -q batch

cd $PBS_O_WORKDIR
mpirun --hostfile $PBS_NODEFILE ./round_trip