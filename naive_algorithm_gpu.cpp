//Written by Hakan Uskan
//AMD HIP code that calculates the number of triangles in any graph on the GPU.
//It works by adjacency matrix representation of a graph.

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Device function to check if three nodes form a triangle
__device__ bool is_triangle(const bool* graph, int n, int a, int b, int c) {
    return graph[a * n + b] && graph[b * n + c] && graph[c * n + a];
}

// Kernel function to count triangles in the graph
__global__ void count_triangles_kernel(const bool* graph, int n, int* count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Calculate the global thread ID
    int totalCombinations = n * (n - 1) * (n - 2) / 6; // Total number of unique triangle combinations

    // Ensure that the thread ID is within the range of possible combinations
    if (tid < totalCombinations) {
        int k = tid; // Local variable to find the combination for this thread
        int a, b, c; // Indices of the nodes to form a triangle

	// Calculate the first node 'a' of the triangle
        // Iterate until we find the correct 'a' such that the remaining combinations are enough for remaining 'k'
        for (a = 0; a < n - 2; a++) {
            if (k < (n - 1 - a) * (n - 2 - a) / 2) {
                break; // Found the correct 'a'
            }
            k -= (n - 1 - a) * (n - 2 - a) / 2; // Adjust 'k' to skip combinations with previous 'a's
        }

	// Calculate the second node 'b' of the triangle, starting from 'a + 1'
        // Iterate to find 'b' such that the remaining 'k' fits the combinations from 'b' to 'n - 1'
        for (b = a + 1; b < n - 1; b++) {
            if (k < (n - 1 - b)) {
                break; // Found the correct 'b'
            }
            k -= (n - 1 - b); // Adjust 'k' for the next 'b'
        }

        c = b + 1 + k; // Calculate the third node 'c' directly using the remaining 'k'

	// Check if the nodes 'a', 'b', and 'c' form a triangle and increment the count atomically if they do
        if (is_triangle(graph, n, a, b, c)) {
            atomicAdd(count, 1);
        }
    }
}

int count_triangles(const bool* graph, int n) {
    bool* d_graph;
    int* d_count;
    int count = 0;
    
    // Allocate memory on the device
    hipMalloc(&d_graph, n * n * sizeof(bool));
    hipMalloc(&d_count, sizeof(int));
    
    // Copy data from host to device
    hipMemcpy(d_graph, graph, n * n * sizeof(bool), hipMemcpyHostToDevice);
    hipMemcpy(d_count, &count, sizeof(int), hipMemcpyHostToDevice);

    // Calculate the number of blocks and threads per block
    int threadsPerBlock = 256;
    int blocks = (n * (n - 1) * (n - 2) / 6 + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    hipLaunchKernelGGL(count_triangles_kernel, blocks, threadsPerBlock, 0, 0, d_graph, n, d_count);

    // Copy the result back to host
    hipMemcpy(&count, d_count, sizeof(int), hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_graph);
    hipFree(d_count);

    return count;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing
    
    const int n = 15; // Number of nodes in the graph
    bool graph[n * n] = {
        0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
	1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, // Graph data here...
	0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0 

    };

    // Count triangles in the graph
    int triangle_count = count_triangles(graph, n);
    std::cout << "Number of triangles: " << triangle_count << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::cout<<"This calculation took '"<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"' milliseconds."<<std::endl;
    return 0;
}

