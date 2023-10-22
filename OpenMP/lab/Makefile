all:
		gcc et.c -fopenmp -o et -lm
		gcc -DPARALLEL=1 et.c -fopenmp -o et_par -lm
		mpicc 1z.c -o 1 -lm
		mpicc -DPARALLEL=1 1z.c -o 1_par -lm
		gcc 2e.c -fopenmp -o 2 -lm
		gcc -DPARALLEL=1 2e.c -fopenmp -o 2_par -lm

check: check_et check_1 check_2

check_et:
		./et 1 et.txt
		./et_par 6 et_par.txt
		diff et.txt et_par.txt
		rm et.txt et_par.txt

check_1:
		mpirun -np 1 ./1 1.txt
		mpirun -np 6 ./1_par 1_par.txt
		diff 1.txt 1_par.txt
		rm 1.txt 1_par.txt

check_2:
		./2 1 2.txt
		./2_par 6 2_par.txt
		diff 2.txt 2_par.txt
		rm 2.txt 2_par.txt
