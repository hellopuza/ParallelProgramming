C=gcc
FLAGS= -fopenmp
CFLAGS=-c $(FLAGS) $(DEFS)
LDFLAGS= $(FLAGS)
SOURCES=main.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=test_soting

all: $(SOURCES) $(EXECUTABLE) clean lib
		
$(EXECUTABLE): $(OBJECTS) 
		$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
		$(CC) $(CFLAGS) $< -o $@

clean:
		rm -rf *.o

lib:
		$(CC) -fPIC $(FLAGS) $(DEFS) -shared -o lib.so $(SOURCES)
