


CC = mpicc
CFLAGS = -std=c99 -g -O3
LIBS = -lm

BIN = matmul

all: $(BIN)

matmul: matmul.c matmul.h
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) -lgslcblas

clean:
	$(RM) $(BIN)
