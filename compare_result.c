#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void print_usage_and_exit(char **argv) {
    fprintf(stderr, "Usage: %s <result0> <result1>\n", argv[0]);
    fprintf(stderr, " e.g., %s result.out answer.out\n", argv[0]);
    exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        print_usage_and_exit(argv);
    }

    FILE *f0 = fopen(argv[1], "r");
    if (!f0) {
        fprintf(stderr, "%s doesn't exist.\n", argv[1]);
        exit(EXIT_FAILURE);
    }
    FILE *f1 = fopen(argv[2], "r");
    if (!f1) {
        fprintf(stderr, "%s doesn't exist.\n", argv[2]);
        exit(EXIT_FAILURE);
    }

    int same = 1;
    while (1) {
        int n;
        char l0[16], l1[16];
        float c0, c1;
        if (fscanf(f0, "Image %04d: %s %f\n", &n, l0, &c0) != 3) break;
        if (fscanf(f1, "Image %04d: %s %f\n", &n, l1, &c1) != 3) break;
        if (strcmp(l0, l1) != 0) {
            printf("Image %04d: different class (%s vs %s)\n", n, l0, l1);
            same = 0;
            break;
        }
        if (fabs(c0 - c1) > 0.01) {
            printf("Image %04d: different confidence (%f vs %f)\n",
                    n, c0, c1);
            same = 0;
            break;
        }
    }
    if (same) {
        printf("Results are same.\n");
    }

    fclose(f0);
    fclose(f1);
    return 0;
}
