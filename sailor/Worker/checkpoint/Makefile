CC=g++
CFLAGS=-pthread -I/usr/local/cuda-12.6/include/ -L/usr/local/cuda-12.6/lib64/ -latomic -lcudart -O3 -march=native -mtune=native -ffast-math

pccheck_writer.o: pccheck_writer.cc
	$(CC) -c -fPIC pccheck_writer.cc -o pccheck_writer.o $(CFLAGS)

libcheck.so: pccheck_writer.o
	$(CC) -shared -o libcheck.so pccheck_writer.o $(CFLAGS)

all: libcheck.so

clean:
	rm -rf *.o *.so