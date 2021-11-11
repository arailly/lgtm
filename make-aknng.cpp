//
// Created by Yusuke Arai on 2021/11/10.
//

#include <iostream>
#include <unistd.h>
#include <cpputil.hpp>
#include <nndescent.hpp>

using namespace std;
using namespace nndescent;

int main(int argc, char** argv) {
    int n, d, K;
    string data_path, save_path;

    opterr = 0;
    int c;
    while ((c = getopt(argc, argv, "n:d:K:o:")) != -1) {
        switch (c) {
            case 'n':
                n = stoi(optarg);
                break;
            case 'd':
                d = stoi(optarg);
                break;
            case 'K':
                K = stoi(optarg);
                break;
            case 'o':
                save_path = optarg;
                break;
            case '?':
                if (optopt == 'n')
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint(optopt))
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf(stderr,
                            "Unknown option character `\\x%x'.\n",
                            optopt);
                return 1;
            default:
                abort();
        }
    }

    if (optind >= argc) {
        fprintf(stderr, "Argument is required.\n");
        return 1;
    }

    data_path = argv[optind];

    AKNNG aknng(n, d, K);
    aknng.build(data_path);
    aknng.save(save_path);
}
