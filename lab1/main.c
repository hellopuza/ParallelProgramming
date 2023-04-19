#include "utils.h"

float f(float t, float x)
{
    return sinf(t) * x * x;
}

float phi(float x)
{
    return sinf(x);
}

float psi(float t)
{
    return t;
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    if (argc < 6)
    {
        printf("Input required\n");
        MPI_Finalize();
        return 1;
    }

    int x_points_num = 0;
    int t_points_num = 0;
    float max_x = 0.0F;
    float max_t = 0.0F;
    int iterations = 0;
    sscanf(argv[1], "%d", &x_points_num);
    sscanf(argv[2], "%d", &t_points_num);
    sscanf(argv[3], "%f", &max_x);
    sscanf(argv[4], "%f", &max_t);
    sscanf(argv[5], "%d", &iterations);

    if (iterations == 0)
    {
        print_solution(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t);
    }
    else
    {
        test_calculation_time(mpi, f, phi, psi, x_points_num, t_points_num, max_x, max_t, iterations);
    }

    MPI_Finalize();
    return 0;
}