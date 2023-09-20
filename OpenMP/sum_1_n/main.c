#include <omp.h>
#include <stdio.h>

float getSum(int begin, int end)
{
    float sum = 0;
    for (int i = begin; i <= end; i++)
    {
        sum += 1.0F / (float)i;
    }
    return sum;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Input required\n");
        return 1;
    }

    float sum = 0.0F;
    int N = 0;
    sscanf(argv[1], "%d", &N);

    #pragma omp parallel reduction(+:sum)
    {
        int threads_num = omp_get_num_threads();
        int current_thread = omp_get_thread_num();

        int range = N / threads_num;
        int begin = current_thread * range + 1;
        int end = (current_thread == threads_num - 1) ? N : (current_thread + 1) * range;

        sum += getSum(begin, end);
    }

    printf("%.8f\n", sum);
    return 0;
}
