#include <cstdio>
#include "lbf/common.hpp"

using namespace std;
using namespace lbf;

// dirty but works
int train(int);
int test(void);
int prepare(void);
int run(void);
int runCamera(int arg);


int main(int argc, char **argv) {
    if (argc < 2) {
        LOG("We need an argument");
        return 0;
    }
    if (strcmp(argv[1], "train") == 0) {
        return train(0);
    }
    if (strcmp(argv[1], "resume") == 0) {
        int start_from;
        printf("Which stage you want to resume from: ");
        scanf("%d", &start_from);
        return train(start_from);
    }
    if (strcmp(argv[1], "test") == 0) {
        return test();
    }
    if (strcmp(argv[1], "prepare") == 0) {
        return prepare();
    }
    if (strcmp(argv[1], "run") == 0) {
        return run();
    }
    if (strcmp(argv[1], "camera") == 0){
        runCamera(atoi(argv[2]));
    }
    else {
        LOG("Wrong Arguments.");
    }
    return 0;
}
