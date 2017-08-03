LDFLAGS = -lm -lfftw3

XTRA_CFLAGS = $(shell pkg-config --cflags fftw3)
XTRA_LDFLAGS = -lm $(shell pkg-config --libs fftw3)

all: genref1d

genref1d.o: genref1d.c
	$(CC) -c genref1d.c $(XTRA_CFLAGS)

genref1d: genref1d.o
	$(CC) -o genref1d genref1d.o $(XTRA_LDFLAGS)

clean:
	$(RM) genref1d genref1d.o
