#include "integration.h"

double f(double x)
{
    return sin(1.0 / (1.0 - x));
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Input required\n");
        return 1;
    }

    int nproc = 0;
    double err = 0.0;
    sscanf(argv[1], "%d", &nproc);
    sscanf(argv[2], "%lf", &err);

    double a = 0.0;
    double b = 0.999;

    print_solution(nproc, f, a, b, err);
    return 0;
}