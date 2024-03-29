//Written by Hakan Uskan
//C++ code that calculates the number of triangles in any graph on the CPU.
//It works by adjacency matrix representation of a graph.

#include <iostream>
#include <vector>
#include <chrono>

// Function to check if three nodes form a triangle
bool is_triangle(const std::vector<std::vector<bool>>& graph, int a, int b, int c) {
    return graph[a][b] && graph[b][c] && graph[c][a];
}

// Function to count the number of triangles in the graph
int count_triangles(const std::vector<std::vector<bool>>& graph) {
    int count = 0;
    int n = graph.size(); // Number of vertices in the graph

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            for (int k = j + 1; k < n; ++k) {
                if (is_triangle(graph, i, j, k)) {
                    ++count;
                }
            }
        }
    }

    return count;
}

int main() {
    auto start = std::chrono::high_resolution_clock::now(); // Start timing
    
    // Example graph represented as an adjacency matrix
    std::vector<std::vector<bool>> graph = {
        {0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
	{1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
	{1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0}, 
	{1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1}, 
	{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0}

    };

    std::cout << "Number of triangles: " << count_triangles(graph) << std::endl;
    
    auto end = std::chrono::high_resolution_clock::now(); // End timing
    std::cout<<"This calculation took '"<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"' milliseconds."<<std::endl;
    
    return 0;
}

