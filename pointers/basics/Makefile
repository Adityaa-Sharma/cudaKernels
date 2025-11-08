
GCC=g++
nvcc=nvcc
CFLAGS=-Wall -g -std=c++11

basics: basics.cpp
	$(GCC) $(CFLAGS) -o basics basics.cpp

pointers: pointers.cpp
	$(GCC) $(CFLAGS) -o pointers pointers.cpp

whoami: whoami.cu
	$(nvcc) -o whoami whoami.cu


clean:
	#clean up the build artifacts
	rm -f basics pointers whoami