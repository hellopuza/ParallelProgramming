#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <mpi.h>

typedef struct
{
    int size;
    int rank;
    MPI_Comm comm;
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

    char port_name[MPI_MAX_PORT_NAME];
    MPI_Open_port(MPI_INFO_NULL, port_name);
    printf("server: port_name = %s\n", port_name);

    MPI_Comm client;
    MPI_Publish_name("name", MPI_INFO_NULL, port_name);
    MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &client);

    int recvbuff;
    MPI_Recv(&recvbuff, 1, MPI_INT, 0, 0, client, MPI_STATUS_IGNORE);
    printf("server: recvbuff = %d\n", recvbuff);

    MPI_Unpublish_name("name", MPI_INFO_NULL, port_name);
    MPI_Close_port(port_name);

    MPI_Finalize();
    return 0;
}