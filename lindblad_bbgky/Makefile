CC=gcc
#Change BLAS_OPT to include links to whatever cblas library you have
BLAS_OPT=-lblas
all: lindblad_bbgky.so

%.o: %.c lindblad_bbgky.h
	$(CC) -Wall -fPIC -O3 -g $< `pkg-config --cflags python` -c -o $@

lindblad_bbgky.so: lindblad_bbgky.o bbgkymodule.o
	$(CC) -shared  $^  `pkg-config --libs python` -O3 $(BLAS_OPT) -o $@

indent:
	indent *.c *.h

clean:
	rm -f *.o *.so *~
