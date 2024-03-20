
CC = mpicc
CFLAGS = -std=c99 -g -O3
LIBS = -lm

BIN = CG1

all: $(BIN)

CG1: CG1.c CG1.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

clean:
	$(RM) $(BIN)

