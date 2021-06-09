CXX := nvcc
TARGET := conv
CUDNN_PATH := /usr/local/cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L /usr/local/lib
CXXFLAGS := -arch=sm_35 -std=c++11 -O2

OPENCV_LIBS=`pkg-config --libs opencv4`
OPENCV_INCLUDE=`pkg-config --cflags opencv4`

all: conv

conv: $(TARGET).cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) $(TARGET).cu -o $(TARGET) \
	-lcudnn $(OPENCV_LIBS) $(OPENCV_INCLUDE)

.phony: clean

clean:
	rm $(TARGET) || echo -n ""