#include <hip/hip_runtime.h>
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
    hipMallocManaged(h_graph, node_count * sizeof(Node));

    for (int i = 0; i < node_count; i++) {
        hipMallocManaged(&(*h_graph)[i].cols, col_sizes[i] * sizeof(int));
        hipMemcpy((*h_graph)[i].cols, all_cols[i], col_sizes[i] * sizeof(int), hipMemcpyHostToDevice);
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


__global__ void G2_count_kernel(Node *graph, int numNodes, int totalComb, int *count, int *combination, int *G2_combs) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < totalComb) {

        int indexes = tid * 3;
        int i = combination[indexes] - 1; //for zero indexing
        int j = combination[indexes + 1] -1;
        int k = combination[indexes + 2] -1;

        int ij_connection = areConnectedDevice(graph, i, j);
        int ik_connection = areConnectedDevice(graph, i, k);
        int jk_connection = areConnectedDevice(graph, j, k);

        if (ij_connection && jk_connection && ik_connection) {
            atomicAdd(count, 1);
            G2_combs[indexes] = i;
            G2_combs[indexes + 1] = j;
            G2_combs[indexes + 2] = k;


            
            //printf("G2 i: %d, j: %d, k: %d\n", i, j, k);
        }
    }
}








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
    int h_count = 0;

    Node *d_graph;
    hipMalloc((void**)&d_graph, numNodes * sizeof(Node));
    
    int *d_count;
    hipMalloc((void**)&d_count, sizeof(int));

    // Copy the host graph to the device
    hipMemcpy(d_graph, h_graph, numNodes * sizeof(Node), hipMemcpyDefault);
    hipMemcpy(d_count, &h_count, sizeof(int), hipMemcpyHostToDevice);




    // Generate all combinations

    int n = numNodes;
    //int k = 3;
    int *ans = (int *)malloc(n * sizeof(int));

    int *comb;
    hipError_t hipStatus;
    int totalcombs = n*(n-1)*(n-2)/3/2 ;
    hipStatus = hipMallocManaged(&comb, totalcombs  * 3 * sizeof(int));
    //int num = 0;
    if (hipStatus != hipSuccess) {
        fprintf(stderr, "hipMallocManaged failed: %s\n", hipGetErrorString(hipStatus));
        return 1;
    }
    // gen(ans, n, k, 0, false, comb, &num);

    // // Print elements of comb
    // printf("Elements of comb array:\n");
    // for (int i = 0; i < 10; ++i) {
    //     printf("%d ", comb[i]);
    // }
    // printf("\n");
 

    // free(ans);
    

    // Read from binary file
    std::ifstream infile("3_node_combinations.bin", std::ios::binary);
    infile.read(reinterpret_cast<char*>(comb), totalcombs  * 3* sizeof(int));
    infile.close();

    int *G2_combs;
    hipMallocManaged(&G2_combs, totalcombs  * 3* sizeof(int));



    //Define block and grid sizes
    int totalComb;  // Update this based on your actual combinations
    totalComb = totalcombs;
    int blockSize = 512;
    int gridSize = (totalComb + blockSize - 1) / blockSize;

    printf("[main] Launching kernels for G2\n");
    // Launch the kernel
    G2_count_kernel<<<gridSize, blockSize >>>(d_graph, numNodes, totalComb, d_count, comb, G2_combs);

    hipDeviceSynchronize();

    
    

    // Copy the results back to the host
    hipMemcpy(&h_count, d_count, sizeof(int), hipMemcpyDeviceToHost);

    // Print the result
    printf("[main] G2 Count: %d\n", h_count);

    int *h_G2_combs =(int *)malloc(h_count *3* sizeof(int));
    //printf("Elements of G1 array:\n");
    int index = 0;
    for (int i = 0; i < n*(n-1)*(n-2)/6; ++i) {
        int idx = i * 3;
        if (G2_combs[idx] || G2_combs[idx+1] || G2_combs[idx+2])
        {
            h_G2_combs[index * 3] = G2_combs[idx];
            h_G2_combs[index * 3 + 1] = G2_combs[idx+1];
            h_G2_combs[index * 3 + 2] = G2_combs[idx+2];
            //printf("index : %d\n", index);
            //printf("i: %d, j: %d, k: %d \n", h_G2_combs[index * 3], h_G2_combs[index * 3 + 1], h_G2_combs[index * 3 + 2]);            
            index++;
        }
    }

    // // Print elements of h_G2_combs
    // printf("Elements of h_G2_combs array:\n");
    // for (int i = 0; i < index; ++i) {
    //     int idx = i * 3;
    //     printf("index : %d\n", idx);
    //     printf("i: %d, j: %d, k: %d \n", h_G2_combs[idx], h_G2_combs[idx+1], h_G2_combs[idx+2]);
    // }
    // printf("\n");


    // Write to binary file
    std::ofstream outfile("G2_node_combinations.bin", std::ios::binary);
    // Write the size at the beginning of the file
    outfile.write(reinterpret_cast<char*>(&h_count), sizeof(int));
    outfile.write(reinterpret_cast<char*>(h_G2_combs), h_count * 3 * sizeof(int));
    outfile.close(); 
    // Free unified memory
    free(h_G2_combs);
    hipFree(comb);
    hipFree(h_graph);
    hipFree(d_graph);
    hipFree(d_count);

    return 0;
}


