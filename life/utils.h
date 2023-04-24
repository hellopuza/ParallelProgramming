#ifndef UTILS_H
#define UTILS_H

#include "graphics.h"
#include "life.h"

comm_info_t init_comm_mpi(MPI_Comm comm)
{
    comm_info_t mpi;
    MPI_Comm_size(comm, &mpi.size);
    MPI_Comm_rank(comm, &mpi.rank);
    mpi.comm = comm;
    return mpi;
}

comm_info_t init_mpi(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    return init_comm_mpi(MPI_COMM_WORLD);
}

void test_simulation_time(comm_info_t mpi, int grid_size, int iterations)
{
    double time = 0.0;

    int8_t* data[2] = {};
    create_grid(data, grid_size);

    for (int i = 0; i < iterations; i++)
    {
        MPI_Barrier(mpi.comm);
        double time0 = MPI_Wtime();

        life_step(mpi, data, grid_size);

        double time1 = MPI_Wtime();
        MPI_Barrier(mpi.comm);

        time += time1 - time0;
    }

    destroy_grid(data);

    if (mpi.rank == 0)
    {
        printf("%.6lf\n", time / iterations);
    }
}

void print_grid(comm_info_t mpi, int grid_size)
{
    int8_t* data[2] = {};
    create_grid(data, grid_size);

    for (int x = 0; x < grid_size; x++)
    {
        for (int y = 0; y < grid_size; y++)
        {
            *at(data[0], grid_size, x, y) = rand() % 2;
        }
    }

    const int window_width = 800;
    const int window_height = 800;

    sf::RenderWindow window(sf::VideoMode(window_width, window_height), "Life", sf::Style::Titlebar | sf::Style::Close);
    if (mpi.rank != 0)
    {
        window.close();
    }

    int is_running = 1;
    while (is_running)
    {
        life_step(mpi, data, grid_size);
        gather_data(mpi, data[0], grid_size);

        if (mpi.rank == 0)
        {
            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                {
                    window.close();
                    is_running = 0;
                    MPI_Bcast(&is_running, 1, MPI_INT, 0, mpi.comm);
                }
            }
            display(&window, data[0], grid_size);
        }
        MPI_Bcast(&is_running, 1, MPI_INT, 0, mpi.comm);
    }

    destroy_grid(data);
}

#endif // UTILS_H