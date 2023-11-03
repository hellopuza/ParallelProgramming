C=
FLAGS= -fopenmp -mavx -mavx2 -march=native
CFLAGS=-c $(FLAGS) $(DEFS)
LDFLAGS= $(FLAGS)
SOURCES=fft.c

all:
		$(CC) -fPIC $(FLAGS) $(DEFS) -shared -o lib.so $(SOURCES) -lm
