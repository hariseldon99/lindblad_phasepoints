CC=gcc
all: main lindblad_bbgky.so

bbgkymodule.o: bbgkymodule.c
	$(CC) -Wall -fPIC -O -g $< `pkg-config --cflags python` -c -o $@

lindblad_bbgky.o: lindblad_bbgky.c 
	$(CC) -Wall -fPIC -O -g $< -c -o $@

lindblad_bbgky.so: lindblad_bbgky.o bbgkymodule.o
	$(CC) -shared  $^  `pkg-config --libs python` -lblas -o $@

main: main.o lindblad_bbgky.o
	$(CC) $^ -lblas -lm -o $@

main.o: main.c 
	$(CC) -Wall -O3 -c $<

clean:
	rm -f main *.o *.so 
