#include <vector>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <mpi.h>

// function to read in a list of 3D coordinates from an .xyz file
// input: the name of the file
std::vector<std::vector<double>> read_xyz_file(std::string filename, int& N, double& L){

    // open the file
    std::ifstream xyz_file(filename);

    // read in the number of atoms
    xyz_file >> N;

    // read in the cell dimension
    xyz_file >> L;

    // now read in the positions, ignoring the atomic species
    std::vector<std::vector<double>> positions;
    std::vector<double> pos = {0, 0, 0};
    std::string dummy; 
    for (int i=0;i<N;i++){
    xyz_file >> dummy >> pos[0] >> pos[1] >> pos[2];
    positions.push_back(pos);           
    }

    // close the file
    xyz_file.close();

    return positions;
}

// Function that finds if two atoms are neighbours given a cutoff distance
bool isNeighbour(std::vector<double> &atom1, std::vector<double> &atom2, double cutoff)
{
    return pow(atom2[0]-atom1[0], 2) + pow(atom2[1]-atom1[1], 2) + pow(atom2[2]-atom1[2], 2) < pow(cutoff, 2);
}

// Function that outputs the number of neighbours for each atom in the input vector
// using a brute force approach and non-load-balanced MPI
std::vector<int> findNeighboursBruteForceMPIv1(std::vector<std::vector<double>>& positions, double L, double cutoff, int rank, int num_procs)
{
    int N = positions.size();
    // Divide number of tasks into chunks
    int start_index = (N / num_procs) * rank;
    int end_index = (N / num_procs) * (rank + 1);
    if (rank == num_procs - 1) 
    {
        end_index = N;
    }

    // Initialise empty results vector
    std::vector<int> results(N, 0);

    // Compute neighbours
    for (int i = start_index; i < end_index; ++i) {
        for (int j = i + 1; j < N; ++j) {
            if (isNeighbour(positions[i], positions[j], cutoff)) {
                results[i]++;
                results[j]++;
            }
        }
    }

    return results;
}

// Function that outputs the number of neighbours for each atom in the input vector
// using a brute force approach and load-balanced MPI
std::vector<int> findNeighboursBruteForceMPIv2(std::vector<std::vector<double>>& positions, double L, double cutoff, int rank, int num_procs)
{
    int N = positions.size();
    // Uses a round-robin approach to balancing the load, instead of chunks.
    std::vector<int> results(N, 0);
    for (int i = rank; i < N; i+=num_procs) {
        for (int j = i + 1; j < N; ++j) {
            if (isNeighbour(positions[i], positions[j], cutoff)) {
                results[i]++;
                results[j]++;
            }
        }
    }

    return results;
}

// Function that outputs the number of neighbours for each atom in the input vector
// using a cell list approach and load-balanced MPI
std::vector<int> findNeighboursCellListMPI(std::vector<std::vector<double>>& positions, double L, double cutoff, int rank, int num_procs)
{
    // Initalise results vector
    std::vector<int> results(positions.size(), 0);

    // Define cell dimensions
    double L_cell = cutoff;

    // Define number of cells in each dimension
    int num_cells = L / L_cell;

    // Create a vector of vectors to store the atoms in each cell
    std::vector<std::vector<int>> cells(num_cells*num_cells*num_cells);

    // Assign each atom to a cell based on its position
    for (int i = 0; i < positions.size(); ++i) 
    {
        // Calculate cell co-ordinates
        int x = (int) (positions[i][0] / L_cell);
        int y = (int) (positions[i][1] / L_cell);
        int z = (int) (positions[i][2] / L_cell);

        // Handle particles outside of the cube
        if (x >= num_cells) {x= num_cells-1;}
        if (y >= num_cells) {y = num_cells-1;}
        if (z >= num_cells) {z = num_cells-1;}
        if (x < 0) {x = 0;}
        if (y < 0) {y = 0;}
        if (z < 0) {z = 0;}

        // Flattens 3D cell co-ordinates into a linear index
        int cell_index = x + y * num_cells + z * num_cells * num_cells;

        // Populates the cells with their atoms.
        cells[cell_index].push_back(i);
    }

    // Iterate over each cell and check if the atoms in that cell have neighbours
    for (int x = rank; x < num_cells; x += num_procs) {
        for (int y = 0; y < num_cells; ++y) {
            for (int z = 0; z < num_cells; ++z) {

                //std::cout << "Rank: " << rank << " handles cell " << x << " " << y << " " << z << std::endl;

                // Calculate linear index from 3D co-ordinates
                int cell_index = x + y * num_cells + z * num_cells * num_cells;

                // Loop over all atoms in the cell
                for (int i = 0; i < cells[cell_index].size(); ++i) {
                    
                    // Find position of atom from the cell index
                    std::vector<double> atom1 = positions[cells[cell_index][i]];

                    // Loop over neighbouring cells (and current cell)
                    for (int dx = -1; dx <= 1; ++dx) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dz = -1; dz <= 1; ++dz) {

                                // Calculate coordinates of the cell
                                int nx = x + dx;
                                int ny = y + dy;
                                int nz = z + dz;

                                // Check if neighbouring cell is within the box
                                if (nx >= 0 && nx < num_cells && ny >= 0 && ny < num_cells && nz >= 0 && nz < num_cells) 
                                {
                                    // Convert neighbour cell coordinates to linear index
                                    int neighbour_cell_index = nx + ny * num_cells + nz * num_cells * num_cells;

                                    // Loop over atoms in current neighbouring cell
                                    for (int j = 0; j < cells[neighbour_cell_index].size(); ++j)
                                    {
                                        // Ensure atom pair haven't already been counted
                                        if (cells[cell_index][i] < cells[neighbour_cell_index][j])
                                        {
                                            // Find position of neighbour atom
                                            std::vector<double> atom2 = positions[cells[neighbour_cell_index][j]];

                                            // Increment the count if the atoms are neighbours
                                            if (isNeighbour(atom1, atom2, cutoff)) 
                                            {
                                                results[cells[cell_index][i]]++;
                                                results[cells[neighbour_cell_index][j]]++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return results;
}

// Function that outputs the number of neighbours for each atom in the input vector given a cutoff distance
// using an atom-centred cell list approach and load-balanced MPI
std::vector<int> findNeighboursCellListMPIv2(std::vector<std::vector<double>>& positions, double L, double cutoff, int rank, int num_procs)
{
    // Initalise results vector
    std::vector<int> results(positions.size(), 0);

    // Define cell dimensions
    double L_cell = cutoff;

    // Define number of cells in each dimension
    int num_cells = L / L_cell;

    // Create a vector of vectors to store the atoms in each cell
    std::vector<std::vector<int>> cells(num_cells*num_cells*num_cells);

    // Assign each atom to a cell based on its position
    for (int i = rank; i < positions.size(); i += num_procs) 
    {
        // Calculate cell co-ordinates
        int x = (int) (positions[i][0] / L_cell);
        int y = (int) (positions[i][1] / L_cell);
        int z = (int) (positions[i][2] / L_cell);

        // Handle particles outside of the cube
        if (x >= num_cells) {x= num_cells-1;}
        if (y >= num_cells) {y = num_cells-1;}
        if (z >= num_cells) {z = num_cells-1;}
        if (x < 0) {x = 0;}
        if (y < 0) {y = 0;}
        if (z < 0) {z = 0;}

        // Flattens 3D cell co-ordinates into a linear index
        int cell_index = x + y * num_cells + z * num_cells * num_cells;

        // Populates the cells with their atoms.
        cells[cell_index].push_back(i);
    }

    // Loop over all atoms
    for (int i = 0; i < positions.size(); ++i)
    {
        // Find position of atom
        std::vector<double> atom1 = positions[i];

        // Calculate cell index
        int x = (int) (atom1[0] / L_cell);
        int y = (int) (atom1[1] / L_cell);
        int z = (int) (atom1[2] / L_cell);
        if (x >= num_cells) {x= num_cells-1;}
        if (y >= num_cells) {y = num_cells-1;}
        if (z >= num_cells) {z = num_cells-1;}
        if (x < 0) {x = 0;}
        if (y < 0) {y = 0;}
        if (z < 0) {z = 0;}
        int cell_index = x + y * num_cells + z * num_cells * num_cells;

        // Loop over neighbouring cells (and current cell)
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {

                    // Calculate coordinates of the cell
                    int nx = x + dx;
                    int ny = y + dy;
                    int nz = z + dz;

                    // Check if neighbouring cell is within the box
                    if (nx >= 0 && nx < num_cells && ny >= 0 && ny < num_cells && nz >= 0 && nz < num_cells) 
                    {
                        // Convert neighbour cell coordinates to linear index
                        int neighbour_cell_index = nx + ny * num_cells + nz * num_cells * num_cells;

                        // Loop over atoms in current neighbouring cell
                        for (int j = 0; j < cells[neighbour_cell_index].size(); ++j)
                        {
                            // Ensure atom pair haven't already been counted
                            if (i < cells[neighbour_cell_index][j])
                            {
                                // Find position of neighbour atom
                                std::vector<double> atom2 = positions[cells[neighbour_cell_index][j]];

                                // Increment the count if the atoms are neighbours
                                if (isNeighbour(atom1, atom2, cutoff)) 
                                {
                                    results[i]++;
                                    results[cells[neighbour_cell_index][j]]++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return results;
}

// Outputs the statistics
void printResults(const std::vector<int>& results, int N, double L)
{
    // Calculate various statistics
    int min_neighbours = *std::min_element(results.begin(), results.end());
    int max_neighbours = *std::max_element(results.begin(), results.end());
    int sum = std::accumulate(results.begin(), results.end(), 0);
    double average = (double) sum / results.size();

    std::cout << "Average number of neighbours: " << average << std::endl;
    std::cout << "Minimum number of neighbours: " << min_neighbours << std::endl;
    std::cout << "Maximum number of neighbours: " << max_neighbours << std::endl;
}

int main(int argc, char **argv)
{ 
    // Initalise MPI call
    MPI_Init(&argc, &argv);

    // Get ranks
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Read in the xyz coordinates
    int N;
    double L;
    std::vector<std::vector<double>> positions = read_xyz_file("argon10549.xyz", N, L);
    
    // Define cut-off distance
    double cutoff_distance = 8;

    /* Brute Force non-load-balancing approach
    ------------------------------------------*/
    // Start timer
    struct timespec start, finish;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Calculate and output results
    if (rank == 0) std::cout << "Brute Force non-load-balancing approach:" << std::endl;
    std::vector<int> results = findNeighboursBruteForceMPIv1(positions, L, cutoff_distance, rank, num_procs);
    std::vector<int> total_results(N, 0);
    MPI_Reduce(&results[0], &total_results[0], N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {printResults(total_results, N, L);}

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double time_elapsed = (finish.tv_sec - start.tv_sec);
    time_elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    if (rank == 0) {std::cout << "Time taken (s): " << time_elapsed << std::endl;}

    /* Brute Force load-balancing approach
    ------------------------------------------*/
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Calculate and output results
    if (rank == 0) std::cout << "Brute Force load-balancing approach:" << std::endl;
    std::vector<int> results1 = findNeighboursBruteForceMPIv2(positions, L, cutoff_distance, rank, num_procs);
    std::vector<int> total_results1(N, 0);
    MPI_Reduce(&results1[0], &total_results1[0], N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {printResults(total_results1, N, L);}

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double time_elapsed1 = (finish.tv_sec - start.tv_sec);
    time_elapsed1 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    if (rank == 0) {std::cout << "Time taken (s): " << time_elapsed1 << std::endl;}

    /* Cell list load-balanced approach
    ------------------------------------------*/
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Calculate and output results
    if (rank == 0) std::cout << "Cell list load-balanced approach:" << std::endl;
    std::vector<int> results2 = findNeighboursCellListMPI(positions, L, cutoff_distance, rank, num_procs);
    std::vector<int> total_results2(N, 0);
    MPI_Reduce(&results2[0], &total_results2[0], N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {printResults(total_results2, N, L);}

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double time_elapsed2 = (finish.tv_sec - start.tv_sec);
    time_elapsed2 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    if (rank == 0) {std::cout << "Time taken (s): " << time_elapsed2 << std::endl;}
    
    /* Atom-centred cell list load-balanced approach
    ------------------------------------------*/
    // Start timer
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Calculate and output results
    if (rank == 0) std::cout << "Atom-centred cell list load-balanced approach:" << std::endl;
    std::vector<int> results3 = findNeighboursCellListMPIv2(positions, L, cutoff_distance, rank, num_procs);
    std::vector<int> total_results3(N, 0);
    MPI_Reduce(&results3[0], &total_results3[0], N, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {printResults(total_results3, N, L);}

    // End timer
    clock_gettime(CLOCK_MONOTONIC, &finish);
    double time_elapsed3 = (finish.tv_sec - start.tv_sec);
    time_elapsed3 += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    if (rank == 0) {std::cout << "Time taken (s): " << time_elapsed3 << std::endl;}

    // Finalize MPI call
    MPI_Finalize();

    return 0;
}