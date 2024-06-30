#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <unordered_map>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Define Edge structure
struct Edge {
    int row;
    int col;
    double value;
};

// CUDA kernel to update node values
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void update_node_values(int *d_rows, int *d_cols, double *d_node_values, int num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        int row = d_rows[idx];
        int col = d_cols[idx];
        atomicAddDouble(&d_node_values[col], d_node_values[row]);
    }
}

void initialize_node_values(thrust::host_vector<double> &node_values, int num_nodes) {
    for (int i = 0; i < num_nodes; i++) {
        node_values[i] = 1.0;
    }
}

// Function to count the number of nodes with the same value
std::unordered_map<double, int> count_node_values(const thrust::host_vector<double> &node_values) {
    std::unordered_map<double, int> value_counts;
    for (auto value : node_values) {
        value_counts[value]++;
    }
    return value_counts;
}

// Main function
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.mtx.bin> <num_iterations>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    int num_iterations = atoi(argv[2]);

    // Read edges from binary file
    FILE *file = fopen(input_filename, "rb");
    if (!file) {
        perror("Failed to open file for reading");
        return EXIT_FAILURE;
    }

    int rows, cols, nonzeros;
    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);
    fread(&nonzeros, sizeof(int), 1, file);

    std::vector<Edge> edges(nonzeros);
    thrust::host_vector<int> h_rows(nonzeros);
    thrust::host_vector<int> h_cols(nonzeros);
    thrust::host_vector<double> h_values(nonzeros);

    for (int i = 0; i < nonzeros; i++) {
        fread(&edges[i].row, sizeof(int), 1, file);
        fread(&edges[i].col, sizeof(int), 1, file);
        fread(&edges[i].value, sizeof(double), 1, file);
        h_rows[i] = edges[i].row;
        h_cols[i] = edges[i].col;
        h_values[i] = edges[i].value;
    }

    fclose(file);

    // Randomly select edges to create subgraphs
    srand(42); // Set a fixed seed for reproducibility
    std::vector<Edge> subgraph_edges;
    for (int i = 0; i < nonzeros; i++) {
        if (rand() % 2) {
            subgraph_edges.push_back(edges[i]);
        }
    }

    // Create device vectors
    thrust::device_vector<int> d_rows(subgraph_edges.size());
    thrust::device_vector<int> d_cols(subgraph_edges.size());
    for (size_t i = 0; i < subgraph_edges.size(); ++i) {
        d_rows[i] = subgraph_edges[i].row;
        d_cols[i] = subgraph_edges[i].col;
    }
    thrust::device_vector<double> d_node_values(rows);

    std::vector<int> kernel_features;
    kernel_features.push_back(rows); // Initial number of nodes

    // Initialize node values only once
    thrust::host_vector<double> h_node_values(rows);
    initialize_node_values(h_node_values, rows);

    // Create CUDA events for timing
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    // Record start event
    hipEventRecord(start, 0);

    // Perform the algorithm for each iteration
    for (int iter = 0; iter < num_iterations; iter++) {
        d_node_values = h_node_values;

        int threads_per_block = 256;
        int num_blocks = (subgraph_edges.size() + threads_per_block - 1) / threads_per_block;

        update_node_values<<<num_blocks, threads_per_block>>>(
                thrust::raw_pointer_cast(d_rows.data()),
                thrust::raw_pointer_cast(d_cols.data()),
                thrust::raw_pointer_cast(d_node_values.data()),
                subgraph_edges.size()
        );

        hipDeviceSynchronize();

        h_node_values = d_node_values;

        auto value_counts = count_node_values(h_node_values);

        for (const auto &pair : value_counts) {
            kernel_features.push_back(pair.second);
        }
    }

    // Record stop event
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);

    // Calculate elapsed time
    float elapsed_time;
    hipEventElapsedTime(&elapsed_time, start, stop);
    printf("Overall Time: %f ms\n", elapsed_time);

    // Destroy CUDA events
    hipEventDestroy(start);
    hipEventDestroy(stop);

    // Print kernel features
    printf("Kernel features:\n");
    for (auto feature : kernel_features) {
        printf("%d ", feature);
    }
    printf("\n");

    return EXIT_SUCCESS;
}

