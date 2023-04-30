#include "integration.h"

double f(double x)
{
    return sin(1.0 / (1.0 - x));
}

double a = 0.0;
double b = 0.999;

double get_time()
{
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec + now.tv_nsec * 1e-9;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        printf("Input required\n");
        return 1;
    }

    int num_thr = 0;
    double err = 0.0;
    sscanf(argv[1], "%d", &num_thr);
    sscanf(argv[2], "%lf", &err);

    int iter_num = 0;
    if (argc == 4)
    {
        sscanf(argv[3], "%d", &iter_num);
    }

    if (iter_num == 0)
    {
        printf("%.16lf\n", integrate(num_thr, f, a, b, err));
    }
    else
    {
        double time = 0;
        for (int i = 0; i < iter_num; i++)
        {
            double time0 = get_time();
            integrate(num_thr, f, a, b, err);
            double time1 = get_time();

            time += time1 - time0;
        }
        printf("%lf\n", time / (double)iter_num);
    }

    return 0;
}