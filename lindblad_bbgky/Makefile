CC=gcc
all: lindblad_bbgky.so

bbgkymodule.o: bbgkymodule.c
	$(CC) -Wall -fPIC -O -g $< `pkg-config --cflags python` -c -o $@

lindblad_bbgky.o: lindblad_bbgky.c 
	$(CC) -Wall -fPIC -O -g $< -c -o $@

lindblad_bbgky.so: lindblad_bbgky.o bbgkymodule.o
	$(CC) -shared  $^  `pkg-config --libs python` -lblas -o $@

clean:
	rm -f main *.o *.so 
