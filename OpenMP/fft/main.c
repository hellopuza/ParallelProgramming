#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "fft.c"

int main()
{
    __m256 w = _mm256_set_ps(0, PI, 2 * PI, 3 * PI, 4 * PI, 5 * PI, 6 * PI, 7 * PI);
    __m256 c = cos256_ps(w);
    __m256 s = sin256_ps(w);
    for (int i = 0; i < 8; i++)
    {
        printf("%f ", ((float*)&c)[i]);
    } printf("\n");

    for (int i = 0; i < 8; i++)
    {
        printf("%f ", ((float*)&s)[i]);
    } printf("\n");

    return 0;
}
