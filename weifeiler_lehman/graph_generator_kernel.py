import numpy as np
import random


def generate_mtx_file(num_nodes, filename):
    # Initialize a set to keep track of edges to ensure they are unique
    edges = set()

    # Randomly determine the number of edges
    num_edges = random.randint(num_nodes, num_nodes * (num_nodes - 1) // 2)

    while len(edges) < num_edges:
        node1 = random.randint(1, num_nodes)
        node2 = random.randint(1, num_nodes)

        # Ensure the edge is not a self-loop and is unique
        if node1 != node2:
            edge = (min(node1, node2), max(node1, node2))
            if edge not in edges:
                edges.add(edge)

    # Open the file to write
    with open(filename, 'w') as f:
        # Write the header for Matrix Market format
        f.write('% Generated random bidirectional graph\n')

        # Write the dimensions and number of edges
        f.write(f'{num_nodes} {num_nodes} {len(edges) * 2}\n')

        # Write the edges
        for edge in edges:
            f.write(f'{edge[0]} {edge[1]} 1\n')
            f.write(f'{edge[1]} {edge[0]} 1\n')


# Example usage
num_nodes = input("Please specify the number of nodes that you need: ")
num_nodes = int(num_nodes.strip())
filename = 'random_graph_10000.mtx'
generate_mtx_file(num_nodes, filename)
print(f'Random .mtx file with {num_nodes} nodes generated: {filename}')
