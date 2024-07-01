//
// Created by malik on 6/23/24.
//

#include <stdio.h>
#include <stdlib.h>

#define MAX_LINE_LENGTH 1024

void write_binary(const char *filename, int rows, int cols, int nonzeros, int *row_indices, int *col_indices, double *values) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        exit(EXIT_FAILURE);
    }

    // Write the matrix dimensions and number of non-zero elements
    fwrite(&rows, sizeof(int), 1, file);
    fwrite(&cols, sizeof(int), 1, file);
    fwrite(&nonzeros, sizeof(int), 1, file);

    // Write each row, col, and value
    for (int i = 0; i < nonzeros; i++) {
        fwrite(&row_indices[i], sizeof(int), 1, file);
        fwrite(&col_indices[i], sizeof(int), 1, file);
        fwrite(&values[i], sizeof(double), 1, file);
    }

    fclose(file);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.mtx> <output.mtx.bin>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *input_filename = argv[1];
    const char *output_filename = argv[2];

    FILE *input_file = fopen(input_filename, "r");
    if (!input_file) {
        perror("Failed to open input file");
        return EXIT_FAILURE;
    }

    char line[MAX_LINE_LENGTH];
    int rows, cols, nonzeros;

    // Read header
    while (fgets(line, sizeof(line), input_file)) {
        if (line[0] != '%') {
            sscanf(line, "%d %d %d", &rows, &cols, &nonzeros);
            break;
        }
    }

    int *row_indices = (int *)malloc(nonzeros * sizeof(int));
    int *col_indices = (int *)malloc(nonzeros * sizeof(int));
    double *values = (double *)malloc(nonzeros * sizeof(double));

    if (!row_indices || !col_indices || !values) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    // Read matrix data
    for (int i = 0; i < nonzeros; i++) {
        if (fscanf(input_file, "%d %d %lf", &row_indices[i], &col_indices[i], &values[i]) != 3) {
            perror("Failed to read matrix data");
            return EXIT_FAILURE;
        }
        // Convert 1-based index to 0-based index
        row_indices[i]--;
        col_indices[i]--;
    }

    fclose(input_file);

    write_binary(output_filename, rows, cols, nonzeros, row_indices, col_indices, values);

    free(row_indices);
    free(col_indices);
    free(values);

    return EXIT_SUCCESS;
}
