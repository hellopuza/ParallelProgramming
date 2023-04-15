#include "sort.h"

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    test_sorting_time(merge_sort, gen_seq_rnd, 10000, 10000000, 2, 100);

    MPI_Finalize();
    return 0;
}