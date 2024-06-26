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


// __host__ __device__ int totalCombinations(int x, int y) {
//     // int numerator = 1;
//     // int denominator = 1;

//     // for (int i = 0; i < y; ++i) {
//     //     numerator *= (x - i);
//     //     denominator *= (i + 1);
//     // }

//     return 244979536;
// }



__global__ void G1_count_kernel(Node *graph, int numNodes, int totalComb, int *count, int *combination, int *G1_combs, int *sizeG1) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalComb) {
        
        int indexes = tid * 3;
        int i = combination[indexes];
        int j = combination[indexes + 1];
        int k = combination[indexes + 2];
        //printf("i: %d j: %d k: %d\n", i, j, k);
        int ij_connection = areConnectedDevice(graph, i, j);
        int ik_connection = areConnectedDevice(graph, i, k);
        int jk_connection = areConnectedDevice(graph, j, k);

        if ((ij_connection && ik_connection && !jk_connection) ||
            (ik_connection && jk_connection && !ij_connection) ||
            (jk_connection && ij_connection && !ik_connection)) {
            atomicAdd(count, 1);
            
            G1_combs[*sizeG1] = i;
            G1_combs[*sizeG1 + 1] = j;
            G1_combs[*sizeG1 + 2] = k;
            atomicAdd(sizeG1, 3);
            
            //printf("G1 i: %d j: %d k: %d\n", i, j, k);
        }
    }
}

// __global__ void test(Node *graph, int numNodes, int totalComb, int *count) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < totalComb) {
        
//         int combination[3];
//         indexToCombinationDevice(numNodes, 3, 2, combination);

//         int i = combination[0];
//         int j = combination[1];
//         int k = combination[2];

//         printf("i: %d j: %d k: %d\n", i, j, k);
//         //int ij_connection = areConnectedDevice(graph, 100, 26);
//         //int ik_connection = areConnectedDevice(graph, 85, 282);

        
//     }
// }


// Helper function for recursive Gray code based combination generation
void gen(int *ans, int n, int k, int idx, bool rev, int *comb, int *size) {
    
    if (k > n || k < 0) {
        return;
    }
    
    if (n == 0) {
        for (int i = 0; i < idx; i++) {
            if (ans[i]) {
                //printf("%d ", i+1);
                comb[*size] = i + 1;
                (*size)++;
            }
        }
        //printf("\n");
        //printf("size: %d\n", *size);
        return;
    }

    ans[idx] = rev;
    gen(ans, n - 1, k - rev, idx + 1, false, comb, size);
    ans[idx] = !rev;
    gen(ans, n - 1, k - !rev, idx + 1, true, comb, size);

}






int main() {
    // Define the graph in unified memory
    Node *h_graph;
    int numNodes;
    
    // Read the graph from the file
    readGraph("graph.txt", &h_graph, &numNodes);
    printf("[main] Number of nodes: %d\n", numNodes);
    int h_count = 0;

    printf("[main] Allocating memory on the device\n");
    Node *d_graph;
    cudaMalloc((void**)&d_graph, numNodes * sizeof(Node));
    
    int *d_count;
    cudaMalloc((void**)&d_count, sizeof(int));

    // Copy the host graph to the device
    cudaMemcpy(d_graph, h_graph, numNodes * sizeof(Node), cudaMemcpyDefault);
    cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice);




    printf("[main] Generating all combinations\n");
    // Generate all combinations

    int n = numNodes;
    //int k = 3;
    //int *ans = (int *)malloc(n * sizeof(int));

    int *comb;
    cudaError_t cudaStatus;

    cudaStatus = cudaMallocManaged(&comb, n*(n-1)*(n-2)/3/2  * 3 * sizeof(int));
    //int num = 0;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }

    int *G1_combs;
    cudaMallocManaged(&G1_combs, 3736*3* sizeof(int));

    int *G1_size = 0;
    cudaMallocManaged(&G1_size, sizeof(int));
    // gen(ans, n, k, 0, false, comb, &num);

    // // Print elements of comb
    // printf("Elements of comb array:\n");
    // for (int i = 0; i < 10; ++i) {
    //     printf("%d ", comb[i]);
    // }
    // printf("\n");

    // // Write to binary file
    // std::ofstream outfile("3_node_combinations.bin", std::ios::binary);
    // outfile.write(reinterpret_cast<char*>(comb), n*(n-1)*(n-2)/3/2  * 3* sizeof(int));
    // outfile.close();    


    


    // // Print elements of comb
    // printf("Elements of comb array:\n");
    // for (int i = 0; i < 10; ++i) {
    //     printf("%d ", comb[i]);
    // }
    // printf("\n");

    //free(ans);
    

    // // Read from binary file
    std::ifstream infile("3_node_combinations.bin", std::ios::binary);
    infile.read(reinterpret_cast<char*>(comb), n*(n-1)*(n-2)/3/2  * 3* sizeof(int));
    infile.close();

    //Define block and grid sizes
    int totalComb;  // Update this based on your actual combinations
    totalComb = 244979536;
    int blockSize = 512;
    int gridSize = (totalComb + blockSize - 1) / blockSize;

    printf("[main] Launching kernel\n");
    // Launch the kernel
    G1_count_kernel<<<gridSize, blockSize >>>(d_graph, numNodes, totalComb, d_count, comb, G1_combs, G1_size);

    cudaDeviceSynchronize();

    // printf("Elements of comb array:\n");
    // for (int i = 0; i < 10; ++i) {
    //     printf("%d ", G1_combs[i]);
    // }
    // printf("\n");

    // Write to binary file
    std::ofstream outfile("G1_node_combinations.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(G1_combs), 3736 * 3* sizeof(int));
    outfile.close(); 

    // Copy the results back to the host
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    printf("[main] Count: %d\n", h_count);

    // Free unified memory
    cudaFree(G1_size);
    cudaFree(G1_combs);
    cudaFree(comb);
    cudaFree(h_graph);
    cudaFree(d_graph);
    cudaFree(d_count);

    return 0;
}

