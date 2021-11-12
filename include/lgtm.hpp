//
// Created by Yusuke Arai on 2021/11/11.
//

#ifndef LGTM_LGTM_HPP
#define LGTM_LGTM_HPP

#include <lsh.hpp>
#include <graph.hpp>

namespace lgtm {
    struct SearchResult {
        time_t time = 0;
        cpputil::Neighbors result;
        unsigned long n_dist_calc = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        void push_back(const SearchResult& result) {
            results.push_back(result);
        }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "time,n_dist_calc";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id,dist";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                line = to_string(result.time) + "," +
                        to_string(result.n_dist_calc);
                log_ofs << line << endl;

                for (const auto& neighbor : result.result) {
                    line = to_string(query_id) + "," +
                           to_string(neighbor.id) + "," +
                           to_string(neighbor.dist);
                    result_ofs << line << endl;
                }

                query_id++;
            }
        }
    };

    struct LGTMIndex {
        int n;
        int dim;
        int n_thread;
        lsh::LSHIndex lsh;
        graph::GraphIndex graph;
        mt19937 engine;

        LGTMIndex(int n_data, int dim, int m, int r, int L, int init_degree) :
                n(n_data),
                dim(dim),
                n_thread(L),
                lsh(n_data, dim, m, r, L),
                graph(n_data, dim, init_degree),
                engine(42) {}

        void build(const string& data_path, const string& graph_path) {
            lsh.build(data_path);
            cout << "complete: build lsh" << endl;

            graph.load(data_path, graph_path, graph.init_degree);
            cout << "complete: load graph" << endl;

            graph.make_bidirectional();
            graph.optimize_edge();
            cout << "complete: optimize graph" << endl;
        }

        auto load(const string& data_path, const string& graph_path) {
            lsh.build(data_path);
            cout << "complete: build lsh" << endl;

            graph.load(data_path, graph_path, graph.max_degree);
            cout << "complete: load graph" << endl;
        }

        auto get_random_ids(int n_id) {
            uniform_int_distribution<> dist(0, n - 1);
            vector<int> random_ids;
            for (int j = 0; j < n_id; ++j)
                random_ids.emplace_back(dist(engine));
            return random_ids;
        }

        auto aknn_search_single(DataArray::Data query, int k,
                                int n_start_node, int ef) {
            auto result = SearchResult();
            const auto start_time = get_now();

            // lsh
            auto start_ids = lsh.find(query, n_start_node);
            if (start_ids.empty()) start_ids.emplace_back(0);

            // graph
            const auto graph_start_time = get_now();
            auto graph_result = graph.knn_search(query, k, ef,
                                                 start_ids, n_start_node);

            result.result = graph_result.result;
            result.n_dist_calc = graph_result.n_dist_calc;

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto aknn_search(DataArray::Data query, int k,
                         int n_start_node, int ef) {
            auto result = SearchResult();
            const auto start_time = get_now();

            vector<graph::SearchResult> graph_results(lsh.L);
#pragma omp parallel for num_threads(n_thread) schedule(dynamic, 1)
            for (int i = 0; i < n_thread; ++i) {
                // lsh
                const auto& hash_table = lsh.hash_tables[i];
                const auto key = lsh.G[i](query);

                try {
                    const auto& start_ids = hash_table.at(key);
                    graph_results[i] = graph.knn_search(query, k, ef, start_ids, n_start_node);
                } catch (out_of_range) {
                    const auto start_ids = vector<int>{0};
                    graph_results[i] = graph.knn_search(query, k, ef, start_ids, n_start_node);
                }
            }

            // merge
            const auto merge_start_time = get_now();

            vector<bool> added(lsh.n);
            for (const auto& graph_result : graph_results) {
                for (const auto& neighbor : graph_result.result) {
                    if (added[neighbor.id]) continue;
                    added[neighbor.id] = true;
                    result.result.emplace_back(neighbor);
                }

                result.n_dist_calc = max(result.n_dist_calc, graph_result.n_dist_calc);
            }

            sort_neighbors(result.result);
            result.result.resize(k);

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);

            return result;
        }
    };
}

#endif //LGTM_LGTM_HPP
