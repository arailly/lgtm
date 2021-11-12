//
// Created by Yusuke Arai on 2021/11/11.
//

#include <iostream>
#include <cpputil.hpp>
#include <unistd.h>
#include <lgtm.hpp>

using namespace std;
using namespace lgtm;

int main(int argc, char** argv) {
    int k, n_query, n, dim, m, t, degree, candidate_size, ns;
    float w;
    string query_path, data_path, index_path;
    string log_path = "log.csv";
    string save_path = "result.csv";

    opterr = 0;
    int c;
    while ((c = getopt(argc, argv, "k:N:n:d:m:w:t:D:c:s:")) != -1) {
        switch (c) {
            case 'k':
                k = stoi(optarg);
                break;
            case 'N':
                n_query = stoi(optarg);
                break;
            case 'n':
                n = stoi(optarg);
                break;
            case 'd':
                dim = stoi(optarg);
                break;
            case 'm':
                m = stoi(optarg);
                break;
            case 'w':
                w = stof(optarg);
                break;
            case 't':
                t = stoi(optarg);
                break;
            case 'D':
                degree = stoi(optarg);
                break;
            case 'c':
                candidate_size = stoi(optarg);
                break;
            case 's':
                ns = stoi(optarg);
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

    if (argc - optind != 3) {
        fprintf(stderr, "3 arguments are required.\n");
        return 1;
    }

    query_path = argv[optind];
    data_path = argv[optind + 1];
    index_path = argv[optind + 2];

    auto index = lgtm::LGTMIndex(n, dim, m, w, t, degree);
    index.load(data_path, index_path);

    auto queries = DataArray(n_query, dim);
    queries.load(query_path);

    lgtm::SearchResults results;
    for (int query_id = 0; query_id < n_query; ++query_id) {
        const auto query = queries.find(query_id);
        auto result = index.aknn_search(query, k, ns, candidate_size);
        results.push_back(move(result));
    }

    results.save(log_path, save_path);
}