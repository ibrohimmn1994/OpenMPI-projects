


CC = mpicc
CFLAGS = -std=c99 -g -O3
LIBS = -lm
BIN = quicksort

all: $(BIN)

quicksort: quicksort.c quicksort.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) $(BIN)

