#include <stdio.h>
#include <nvtx3/nvToolsExt.h>

int main() {
    // Push a range with the name "my_range"
    nvtxRangePush("my_range");

    // Do something here that you want to profile

    // Pop the range we pushed earlier
    nvtxRangePop();

    // Print a message to indicate the program has completed
    printf("Program completed.\n");

    return 0;
}
