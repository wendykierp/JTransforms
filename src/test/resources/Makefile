fftwdir = "E:\devel\fftw-3.2.2"

CC = gcc
CFLAGS = -I$(fftwdir)
LDFLAGS = -L$(fftwdir) -lm -lfftw3-3

all: genref1d
genref1d.o: genref1d.c
	gcc -c genref1d.c $(CFLAGS)
genref1d: genref1d.o
	gcc -o genref1d genref1d.o $(LDFLAGS)