#include "utils.h"

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    if (argc < 3)
    {
        printf("Input required\n");
        MPI_Finalize();
        return 1;
    }

    int grid_size = 0;
    int iterations = 0;
    sscanf(argv[1], "%d", &grid_size);
    sscanf(argv[2], "%d", &iterations);

    if (iterations == 0)
    {
        print_grid(mpi, grid_size);
    }
    else
    {
        test_simulation_time(mpi, grid_size, iterations);
    }

    MPI_Finalize();
    return 0;
}