//
// Created by Yusuke Arai on 2021/11/10.
//

#include <iostream>
#include <cpputil.hpp>
#include <unistd.h>
#include <graph.hpp>

using namespace std;
using namespace graph;

int main(int argc, char** argv) {
    int n, d, degree;
    string data_path, aknng_path, save_path;

    opterr = 0;
    int c;
    while ((c = getopt(argc, argv, "n:d:D:o:")) != -1) {
        switch (c) {
            case 'n':
                n = stoi(optarg);
                break;
            case 'd':
                d = stoi(optarg);
                break;
            case 'D':
                degree = stoi(optarg);
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

    if (argc - optind != 2) {
        fprintf(stderr, "2 arguments are required.\n");
        return 1;
    }

    data_path = argv[optind];
    aknng_path = argv[optind + 1];

    auto index = GraphIndex(n, d, degree);
    index.build(data_path, aknng_path);
    index.save(save_path);
}