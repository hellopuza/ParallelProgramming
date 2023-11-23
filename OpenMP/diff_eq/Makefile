C=
FLAGS= -fopenmp -march=native
CFLAGS=-c $(FLAGS) $(DEFS)
LDFLAGS= $(FLAGS)
SOURCES=diff.c

all:
		$(CC) -fPIC $(FLAGS) $(DEFS) -shared -o lib.so $(SOURCES) -lm
