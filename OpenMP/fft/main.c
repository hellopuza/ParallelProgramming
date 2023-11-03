#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "fft.c"

int main()
{
    int size = 32;
    float* in = (float*)malloc(sizeof(float) * size);
    complex_t* out = (complex_t*)malloc(sizeof(complex_t) * size);
    for (int i = 0; i < size; i++)
    {
        in[i] = i;
    }
    fft_parallel(in, out, size);
    fft_parallel_opt(in, out, size);

    return 0;
}
