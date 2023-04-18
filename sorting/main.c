#define TEST_SORTING_CORRECTNESS 0

#include "sort.h"

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    if (argc < 6)
    {
        printf("Input required\n");
        MPI_Finalize();
        return 1;
    }

    int mode = 0;
    int initial_size = 0;
    int end_size = 0;
    float factor = 0.0F;
    int iterations = 0;

    sscanf(argv[1], "%d", &mode);
    sscanf(argv[2], "%d", &initial_size);
    sscanf(argv[3], "%d", &end_size);
    sscanf(argv[4], "%f", &factor);
    sscanf(argv[5], "%d", &iterations);

    switch (mode)
    {
    case 0:
        test_sorting_time(mpi, merge_sort, gen_seq_rnd, initial_size, end_size, factor, iterations);
        break;
    case 1:
        test_sorting_time(mpi, merge_sort, gen_seq_worst, initial_size, end_size, factor, iterations);
        break;
    case 2:
        test_sorting_time(mpi, merge_sort, gen_seq_best, initial_size, end_size, factor, iterations);
        break;
    case 3:
        test_sorting_time(mpi, merge_sort_parallel, gen_seq_rnd, initial_size, end_size, factor, iterations);
        break;
    case 4:
        test_sorting_time(mpi, merge_sort_parallel, gen_seq_worst, initial_size, end_size, factor, iterations);
        break;
    case 5:
        test_sorting_time(mpi, merge_sort_parallel, gen_seq_best, initial_size, end_size, factor, iterations);
        break;
    }

    MPI_Finalize();
    return 0;
}