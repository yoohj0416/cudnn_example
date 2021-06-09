#define IN_SIZE (1*1*5*5)
#define OUT_SIZE (1*1*3*3)
#define TOL (0.000005)

#define CUDNN_DTYPE CUDNN_DATA_FLOAT
typedef float stype;
typedef float dtype;

dtype input[IN_SIZE] = {0.50f, 0.12f, 0.23f, 0.73f, 0.99f, 0.07f, 0.87f, 0.17f, 0.36f, 0.63f, 0.45f, 0.36f, 0.28f, 0.89f, 0.76f, 0.27f, 0.43f, 0.12f, 0.09f, 0.92f, 0.83, 0.54f, 0.61f, 0.44f, 0.59f};

dtype output[OUT_SIZE] = {0.87f, 0.89f, 0.99f, 0.87f, 0.89f, 0.92f, 0.83f, 0.89f, 0.92f};