C=gcc
CFLAGS=-c -fopenmp $(DEFS)
LDFLAGS= -fopenmp
SOURCES=main.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=ser_access

all: $(SOURCES) $(EXECUTABLE) clean
		
$(EXECUTABLE): $(OBJECTS) 
		$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
		$(CC) $(CFLAGS) $< -o $@

clean:
		rm -rf *.o
