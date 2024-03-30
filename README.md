# CPU-GPU_CountingTrianglesInGraphs
A Naive Algorithm for counting graphlets of size 3 (triangles) in graphs.
-------------------------------------------------------------------------
In this repo you can find a naive algorithm for counting graphlets (small connected subgraphs) in a graph typically involves checking all possible combinations of vertices to find all subgraphs of a certain size and then determining which of these subgraphs are graphlets. For this example, let's consider counting graphlets of size 3 (triangles) in a small graph.

Here's a simple pseudocode algorithm to count triangles (3-node graphlets) in a graph:

1. Initialize a count variable to zero.
2. For each combination of three nodes in the graph:
   a. Check if these three nodes form a triangle.
   b. If they do, increment the count.
3. Return the count.

**CPU**
---
Here is an implementation of a naive algorithm in C++ for counting triangles (3-node graphs) in a graph. This example uses an adjacency matrix to represent the graph, simplifying the check for edges between nodes' existence.

`naive_algorithm_cpu.cpp`

In this code:

- The `graph` is represented as a 2D vector of `bool`, where `graph[i][j]` is `true` if there is an edge between nodes `i` and `j`, and `false` otherwise.
- The `is_triangle` function checks if nodes `a`, `b`, and `c` form a triangle.
- The `count_triangles` function counts the number of triangles in the graph by checking every combination of three nodes.

This algorithm iterates through all combinations of three nodes, making its time complexity ***O(n^3)*** for a graph with ***n*** vertices.

Compilde command:
`g++ -o naive_algorithm_cpu naive_algorithm_cpu.cpp`

Run command:
`./naive_algorith_cpu`

**GPU**
---
Implementing a parallel algorithm for counting triangles in a graph using AMD HIP requires writing a kernel function that can execute in parallel across multiple GPU threads. In addition, writing and executing HIP code requires an appropriate AMD GPU environment. The parallel approach typically involves distributing the work of checking different combinations of nodes across multiple threads. Here’s a conceptual outline of how you might implement it in AMD HIP:

1. Each GPU thread checks a different set of nodes for forming a triangle.
2. Use atomic operations to safely increment the count of triangles.

Here is an implementation of a naive algorithm using parallel programming on GPU in AMD HIP to count triangles (3-node graphs) in a graph. This example uses an adjacency matrix to represent the graph, simplifying the check for edges between nodes' existence.

`naive_algorithm_gpu.cpp`

This code counts the number of triangles in a given graph using GPU parallel processing with AMD's HIP framework. A triangle in a graph is defined as a set of three nodes where each node is connected to the other two by edges. To execute this HIP code, you would need an AMD GPU setup with the appropriate software and drivers installed, and you would compile the code using the HIP compiler. To compile and run the HIP code on an AMD GPU, you need to have the ROCm platform installed, which includes the HIP compiler hipcc. Here's how you can compile the HIP code:

1. Save your HIP source code to a file, say `naive_algotith_gpu.cpp`.
2. Use the `hipcc` compiler to compile your code. Open a terminal and navigate to the directory containing your source file.
3. Compile the code using the following command:
   `hipcc naive__algorith_gpu.cpp -o naive_algorithm_gpu`

This command compiles the `naive_algorith_gpu.cpp` file and creates an executable named `naive_algorith`.

To run the compiled program, simply execute it from the terminal:
`./naive_algorith_gpu`

Make sure you have the ROCm platform properly installed and configured on your system with an AMD GPU that supports HIP. For detailed instructions on installing ROCm and setting up your environment for HIP development, you should refer to the official ROCm documentation: [ROCm Installation](https://rocm.docs.amd.com/en/latest/).

If you encounter any errors during compilation or execution, the error messages should give you an indication of what went wrong, which could be related to the code, the ROCm installation, or the compatibility of your hardware with ROCm and HIP.

In summary, this code leverages the parallel processing power of a GPU to efficiently count the number of triangles in a graph, which is a common problem in various fields like network analysis, social media analytics, and bioinformatics. The use of parallel processing aims to speed up the computation, especially beneficial for large graphs where the number of potential triangles can be very large.

**Important Additional Information**
--------------------------------

Parallel programming, especially on GPUs with frameworks like AMD HIP or NVIDIA CUDA, can sometimes take more time than sequential CPU code for several reasons, especially for small-scale problems or non-optimized implementations:

1. **Overhead:** Parallel programming involves overheads such as memory allocation on the GPU, data transfer between CPU and GPU, and kernel launch overhead. For small problems, the time saved by parallel execution can be less than the overhead introduced, leading to a longer total execution time compared to a CPU-only implementation.

2. **Utilization:** If the problem size is too small, it may not utilize the full parallel processing capabilities of the GPU. GPUs are designed to handle large-scale data parallelism efficiently. If the workload is too small, many GPU cores may remain idle, not providing the expected performance boost.

3. **Memory Access Patterns:** GPUs are optimized for certain memory access patterns to exploit their memory hierarchy effectively. Non-optimal access patterns in GPU code can lead to memory bottlenecks, reducing the performance benefits of parallel execution.

4. **Algorithm Complexity:** Not all algorithms benefit equally from parallelization. Some algorithms have inherently sequential parts that can become bottlenecks in parallel execution. The performance gain from parallelization depends heavily on the algorithm’s ability to be parallelized and the proportion of the code that can run in parallel (Amdahl’s Law).

5. **Synchronization and Atomic Operations:** Parallel code often requires synchronization mechanisms (like barriers or atomic operations) to ensure correct execution. These mechanisms can introduce delays, especially if many threads contend for the same resources or if there is a significant amount of work serialized through atomic operations.

For these reasons, while parallel programming can significantly reduce execution time for large-scale problems or highly parallelizable algorithms, it may not always be faster for smaller problems or algorithms that do not parallelize well. It's important to consider the problem size, data transfer overheads, and the parallelizability of the algorithm when deciding to use parallel programming approaches. Optimization of parallel code (like minimizing data transfer, optimizing memory access patterns, and ensuring high utilization of the GPU cores) is crucial to achieving performance improvements.

*Efforts to run the algorithm with different graph inputs and increase efficiency are continuing...*
