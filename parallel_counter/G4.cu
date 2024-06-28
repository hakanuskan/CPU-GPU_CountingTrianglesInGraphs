#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>

typedef struct Node {
    int row;
    int *cols;
    int cols_size;
    struct Node *next;
} Node;

void readGraph(const char *filename, Node **h_graph, int *numNodes) {
    std::ifstream file(filename);
    std::string line;
    Node *nodes = nullptr;
    int **all_cols = nullptr;
    int *col_sizes = nullptr;
    int node_count = 0;

    while (std::getline(file, line)) {
        node_count++;
    }

    file.clear();
    file.seekg(0);

    nodes = (Node *)malloc(node_count * sizeof(Node));
    all_cols = (int **)malloc(node_count * sizeof(int *));
    col_sizes = (int *)malloc(node_count * sizeof(int));

    int current_node = 0;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string part;
        Node node;

        // Read node id (row)
        std::getline(ss, part, '/');
        node.row = std::stoi(part);

        // Read connected nodes
        std::getline(ss, part, '/');
        part = part.substr(1, part.size() - 2); // Remove '[' and ']'
        std::stringstream ss_cols(part);
        std::string col;
        int col_count = 0;

        while (std::getline(ss_cols, col, ',')) {
            col_count++;
        }

        all_cols[current_node] = (int *)malloc(col_count * sizeof(int));
        ss_cols.clear();
        ss_cols.str(part);

        col_count = 0;
        while (std::getline(ss_cols, col, ',')) {
            all_cols[current_node][col_count++] = std::stoi(col);
        }

        node.cols_size = col_count;
        col_sizes[current_node] = col_count;

        nodes[current_node] = node;
        current_node++;
    }

    *numNodes = node_count;
    cudaMallocManaged(h_graph, node_count * sizeof(Node));

    for (int i = 0; i < node_count; i++) {
        cudaMallocManaged(&(*h_graph)[i].cols, col_sizes[i] * sizeof(int));
        cudaMemcpy((*h_graph)[i].cols, all_cols[i], col_sizes[i] * sizeof(int), cudaMemcpyHostToDevice);
        (*h_graph)[i].cols_size = col_sizes[i];
        (*h_graph)[i].row = nodes[i].row;
        (*h_graph)[i].next = (i < node_count - 1) ? &(*h_graph)[i + 1] : nullptr;
    }

    for (int i = 0; i < node_count; i++) {
        free(all_cols[i]);
    }
    free(all_cols);
    free(col_sizes);
    free(nodes);
}


__device__ int areConnectedDevice(Node *graph, int i, int j) {
    //printf("[areConnectedDevice] Checking if nodes %d and %d are connected\n", i, j);
    Node *current = graph;
    
    
    while (current != nullptr) {
        if (current->row == i) {
            //printf("[areConnectedDevice] Found node %d\n", current->row);
            break;
        }
        current = current->next;
    }

    if (current == nullptr) {
        //printf("[areConnectedDevice] Node %d not found\n", i);
        return 0;
    }


    for (int k = 0; k < current->cols_size; k++) {
        if (current->cols[k] == j) {
            //printf("[areConnectedDevice] Nodes %d and %d are connected\n", i, j);
            return 1;
        }
    }

    //printf("[areConnectedDevice] Nodes %d and %d are not connected\n", i, j);
    return 0;
}



__global__ void G4_count_kernel(Node *graph, int numNodes, int totalComb,int l, int *count, int *combination) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < totalComb) {


        int indexes = tid * 3;
        int i = combination[indexes];
        int j = combination[indexes + 1];
        int k = combination[indexes + 2];
        
        int ij_connection = areConnectedDevice(graph, i, j);
        int ik_connection = areConnectedDevice(graph, i, k);
        int jk_connection = areConnectedDevice(graph, j, k);
        int il_connection = areConnectedDevice(graph, i, l);
        int jl_connection = areConnectedDevice(graph, j, l);
        int kl_connection = areConnectedDevice(graph, k, l);


        if ((ij_connection && jk_connection && jl_connection &&
             !kl_connection && !ik_connection && !il_connection) ||
            (ij_connection && ik_connection && il_connection &&
             !kl_connection && !jk_connection && !jl_connection) ||
            (jk_connection && ik_connection && kl_connection &&
             !il_connection && !ij_connection && !jl_connection) ||
            (jl_connection && il_connection && kl_connection &&
             !ik_connection && !ij_connection && !jk_connection)) {
            
            if (!(i ==0 && j == 0 && k == 0) && l != i && l != j && l != k)
            {
                atomicAdd(count, 1);
                //printf("i: %d, j: %d, k: %d, l: %d\n", i, j, k, l);
                
                
            }
            
            
        }
    }
}









int main() {
    // Define the graph in unified memory
    Node *h_graph;
    int numNodes;
    
    // Read the graph from the file
    readGraph("graph.txt", &h_graph, &numNodes);
    int h_count = 0;

    Node *d_graph;
    cudaMalloc((void**)&d_graph, numNodes * sizeof(Node));
    
    int *d_count;
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy the host graph to the device
    cudaMemcpy(d_graph, h_graph, numNodes * sizeof(Node), cudaMemcpyDefault);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);







    // // Read from binary file
    std::ifstream infile("G1_node_combinations.bin", std::ios::binary);

    // Read the size of the array
    int size_of_G1_combs;
    infile.read(reinterpret_cast<char*>(&size_of_G1_combs), sizeof(int));

    int *comb;
    cudaError_t cudaStatus;

    cudaStatus = cudaMallocManaged(&comb, size_of_G1_combs  * 3 * sizeof(int));
    //int num = 0;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    infile.read(reinterpret_cast<char*>(comb), size_of_G1_combs  * 3 * sizeof(int));
    infile.close();


    // // Print elements of h_G1_combs
    // printf("Elements of h_G1_combs array:\n");
    // for (int i = 0; i < size_of_G1_combs; ++i) {
    //     int idx = i * 3;
    //     printf("index : %d\n", idx);
    //     printf("i: %d, j: %d, k: %d \n", comb[idx], comb[idx+1], comb[idx+2]);
    // }
    // printf("\n");

    //Define block and grid sizes
    int totalComb;  // Update this based on your actual combinations
    totalComb = size_of_G1_combs;
    int blockSize = 512;
    int gridSize = (totalComb + blockSize - 1) / blockSize;

    printf("[main] Launching kernels for G4\n");
    // Launch the kernel
    int l;
    for(l = 0; l < numNodes; l++){
        G4_count_kernel<<<gridSize, blockSize >>>(d_graph, numNodes, totalComb, l ,d_count, comb);
    }

    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("[main] G4 Count: %d\n", h_count/3); // Counted 3 times because of implementation

    // Free unified memory
    cudaFree(comb);
    cudaFree(h_graph);
    cudaFree(d_graph);
    cudaFree(d_count);

    return 0;
}










