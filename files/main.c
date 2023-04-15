#include <mpi.h>
#include <stdio.h>
#include <string.h>

typedef struct
{
    int size;
    int rank;
} comm_info_t;

comm_info_t init_mpi(int argc, char* argv[])
{
    comm_info_t mpi;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
    return mpi;
}

int main(int argc, char* argv[])
{
    comm_info_t mpi = init_mpi(argc, argv);

    MPI_File fp;
    MPI_File_open(MPI_COMM_WORLD, "out.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &fp);

    int buff_size = 2;
    char str[buff_size + 1];
    sprintf(str, "%*d", buff_size, mpi.rank);
    char buff[buff_size];
    memcpy(buff, str, buff_size);

    int offset = buff_size * (mpi.size - mpi.rank - 1);
    MPI_File_write_at(fp, offset, buff, buff_size, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&fp);
    MPI_Finalize();
    return 0;
}