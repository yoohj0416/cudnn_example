all:
	nvcc -arch=sm_35 -std=c++11 -O2 -I /usr/cuda/include -L /usr/cuda/lib64 -L /usr/local/lib pool.cu -o pool -lcudnn -I.
