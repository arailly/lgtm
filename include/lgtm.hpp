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
        time_t lsh_time = 0;
        time_t graph_time = 0;
        time_t merge_time = 0;
        vector<cpputil::Neighbor> result;
        unsigned long n_bucket_content = 0;
        unsigned long n_node_access = 0;
        unsigned long n_dist_calc = 0;
        unsigned long n_hop = 0;
        double dist_from_start = 0;
        double recall = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        void push_back(const SearchResult& result) {
            results.push_back(result);
        }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "time,lsh_time,graph_time,merge_time,"
                          "n_bucket_content,n_node_access,n_dist_calc,"
                          "n_hop,dist_from_start,recall";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id,dist";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                line = to_string(result.time) + "," +
                        to_string(result.lsh_time) + "," +
                        to_string(result.graph_time) + "," +
                        to_string(result.merge_time) + "," +
                        to_string(result.n_bucket_content) + "," +
                        to_string(result.n_node_access) + "," +
                        to_string(result.n_dist_calc) + "," +
                        to_string(result.n_hop) + "," +
                        to_string(result.dist_from_start) + "," +
                        to_string(result.recall);
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

        auto knn_search(DataArray::Data query, int k,
                        int n_start_node, int ef) {
            auto result = SearchResult();
            const auto start_time = get_now();

            // lsh
            auto start_ids = lsh.find(query, n_start_node);
            if (start_ids.empty()) start_ids.emplace_back(0);

            result.n_bucket_content = start_ids.size();
            const auto lsh_end_time = get_now();
            result.lsh_time = get_duration(start_time, lsh_end_time);

            // graph
            const auto graph_start_time = get_now();
            auto graph_result = graph.knn_search(query, k, ef,
                                                 start_ids, n_start_node);

            result.result = graph_result.result;
            result.n_node_access = graph_result.n_node_access;
            result.n_dist_calc = graph_result.n_dist_calc;
            result.n_hop = graph_result.n_hop;
            result.dist_from_start = graph_result.dist_from_start;

            const auto end_time = get_now();
            result.graph_time = get_duration(graph_start_time, end_time);
            result.time = get_duration(start_time, end_time);

            return result;
        }

        auto knn_search_para(DataArray::Data query, int k,
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

                result.lsh_time = max(result.lsh_time, graph_result.lsh_time);
                result.graph_time = max(result.graph_time, graph_result.time);
                result.n_node_access = max(result.n_node_access, graph_result.n_node_access);
                result.n_dist_calc = max(result.n_dist_calc, graph_result.n_dist_calc);
                result.n_hop = max(result.n_hop, graph_result.n_hop);
                result.dist_from_start = max(result.dist_from_start, graph_result.dist_from_start);
            }

            sort_neighbors(result.result);
            result.result.resize(k);

            const auto end_time = get_now();
            result.time = get_duration(start_time, end_time);
            result.merge_time = get_duration(merge_start_time, end_time);

            return result;
        }
    };
}

#endif //LGTM_LGTM_HPP
