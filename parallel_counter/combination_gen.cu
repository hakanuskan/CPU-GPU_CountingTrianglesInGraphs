#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sstream>
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

    int numNodes = 1138;

    printf("[main] Generating all combinations\n");
    // Generate all combinations

    int n = numNodes;
    int k = 3;
    int *ans = (int *)malloc(n * sizeof(int));

    int *comb;
    cudaError_t cudaStatus;

    cudaStatus = cudaMallocManaged(&comb, n*(n-1)*(n-2)/3/2  * 4 * sizeof(int));
    int num = 0;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMallocManaged failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    printf("flag\n");
    //gen(ans, n, k, 0, false, comb, &num);

    // Print elements of comb
    printf("Elements of comb array:\n");
    int a = 0;
    for (int i = 0; i < 210*4; ++i) {
        printf("%d ", comb[i]);
        a += 1;
        if (a % 4 == 0) {
            printf("\n");
        }
    }
    printf("\n");

    // Write to binary file
    std::ofstream outfile("3_node_combinations.bin", std::ios::binary);
    outfile.write(reinterpret_cast<char*>(comb), n*(n-1)*(n-2)/3/2  * 3* sizeof(int));
    outfile.close();    

    free(ans);
    

    // // Read from binary file
    // std::ifstream infile("3_node_combinations.bin", std::ios::binary);
    // infile.read(reinterpret_cast<char*>(comb), n*(n-1)*(n-2)/3/2  * 3* sizeof(int));


    // gen(ans, n, 1, 0, false, comb, &num);
    // infile.close();



}