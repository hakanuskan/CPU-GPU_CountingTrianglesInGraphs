#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>

int getNumNodesFromFile(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Unable to open file\n");
        return -1;
    }

    char line[256];
    char lastLine[256];
    while (fgets(line, sizeof(line), file)) {
        strcpy(lastLine, line);
    }

    fclose(file);

    int firstInt;
    if (sscanf(lastLine, "%d", &firstInt) == 1) {
        return firstInt + 1;
    } else {
        printf("Failed to read the number of nodes from the last line\n");
        return -1;
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
    
    int numNodes = getNumNodesFromFile("graph.txt");
    

    printf("[main] Generating all combinations\n");
    // Generate all combinations

    int n = numNodes;
    int k = 3;
    int *ans = (int *)malloc(n * sizeof(int));
    int totalcombs = n*(n-1)*(n-2)/3/2 ;
    int *comb;
    cudaError_t cudaStatus;

    cudaStatus = cudaMallocManaged(&comb, totalcombs  * 3 * sizeof(int));
    int num = 0;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    gen(ans, n, k, 0, false, comb, &num);

    // // Print elements of comb
    // int index = 0;
    // printf("Elements of comb array:\n");
    // int a = 0;
    // for (int i = 0; i < totalcombs *3 ; ++i) {
    //     printf("%d ", comb[i]);
    //     a += 1;
    //     if (a % 3 == 0) {
    //         index ++;
    //         printf(" -%d \n", index);
    //     }
    // }
    // printf("\n");

    // Write to binary file
    std::ofstream outfile("3_node_combinations.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(comb), totalcombs  * 3* sizeof(int));
    outfile.close();    

    free(ans);
    printf("Number of combinations: %d\n", num);
    printf("Number of nodes: %d\n", n);

    // // Read from binary file
    // std::ifstream infile("3_node_combinations.bin", std::ios::binary);
    // infile.read(reinterpret_cast<char*>(comb), totalcombs  * 3* sizeof(int));


    // gen(ans, n, 1, 0, false, comb, &num);
    // infile.close();



}