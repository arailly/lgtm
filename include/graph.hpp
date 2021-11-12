//
// Created by Yusuke Arai on 2021/11/11.
//

#ifndef LGTM_GRAPH_HPP
#define LGTM_GRAPH_HPP

#include <queue>
#include <cpputil.hpp>

using namespace std;
using namespace cpputil;

namespace graph {
    struct SearchResult {
        vector<Neighbor> result;
        unsigned long n_dist_calc = 0;
    };

    struct GraphIndex {
        int n, dim;
        DataArray dataset;
        vector<Neighbors> edgeset;
        int init_degree, max_degree;

        GraphIndex(int n, int dim, int init_degree) :
                n(n),
                dim(dim),
                dataset(n, dim),
                init_degree(init_degree),
                max_degree(init_degree * 2),
                edgeset(vector<Neighbors>(n)) {}

        auto calc_dist(DataArray::Data data_1,
                       DataArray::Data data_2) {
#ifdef __AVX__
            return l2_sqr_avx(&(*data_1), &(*data_2), dim);
#endif
            float res = 0;
            for (int i = 0; i < dim; ++i, ++data_1, ++data_2) {
                float tmp = *data_1 - *data_2;
                res += tmp * tmp;
            }
            return sqrt(res);
        }

        auto knn_search(DataArray::Data query, int k, int ef,
                        const vector<int>& start_ids, int n_start_id) {
            auto result = SearchResult();

            priority_queue<Neighbor, vector<Neighbor>, CompGreater>
                    candidates;
            priority_queue<Neighbor, vector<Neighbor>, CompLess>
                    top_candidates;

            vector<bool> visited(n);

            Neighbors initial_candidates;

            // calculate distance to start nodes
            n_start_id = min(n_start_id, (int)start_ids.size());
            for (int i = 0; i < n_start_id; ++i) {
                const auto start_id = start_ids[i];
                const auto start_data = dataset.find(start_id);
                const auto dist = calc_dist(query, start_data);

                initial_candidates.emplace_back(dist, start_id);
            }

            // decide nearest node as start node
            sort_neighbors(initial_candidates);
            const auto nearest_start_candidate = initial_candidates[0];
            visited[nearest_start_candidate.id] = true;
            candidates.emplace(nearest_start_candidate);
            top_candidates.emplace(nearest_start_candidate);

            while (!candidates.empty()) {
                const auto nearest_candidate = candidates.top();
                const auto& nearest_candidate_data =
                        dataset.find(nearest_candidate.id);
                candidates.pop();

                if (nearest_candidate.dist > top_candidates.top().dist)
                    break;

                for (const auto& neighbor : edgeset[nearest_candidate.id]) {
                    if (visited[neighbor.id]) continue;
                    visited[neighbor.id] = true;

                    const auto neighbor_data = dataset.find(neighbor.id);
                    const auto dist_from_neighbor =
                            calc_dist(query, neighbor_data);
                    ++result.n_dist_calc;

                    if (top_candidates.size() < ef ||
                        dist_from_neighbor < top_candidates.top().dist) {
                        candidates.emplace(
                                dist_from_neighbor, neighbor.id);
                        top_candidates.emplace(
                                dist_from_neighbor, neighbor.id);

                        if (top_candidates.size() > ef)
                            top_candidates.pop();
                    }
                }
            }

            while (!top_candidates.empty()) {
                result.result.emplace_back(top_candidates.top());
                top_candidates.pop();
            }

            reverse(result.result.begin(), result.result.end());
            if (result.result.size() > k) result.result.resize(k);
            return result;
        }

        auto make_bidirectional() {
            for (int head_id = 0; head_id < n; ++head_id) {
                for (auto& neighbor : edgeset[head_id]) {
                    // add reverse edge
                    auto tail_id = neighbor.id;
                    edgeset[tail_id].emplace_back(neighbor.dist, head_id);
                }
            }
        }

        void optimize_edge() {
            for (int node_id = 0; node_id < n; ++node_id) {
                auto& neighbors = edgeset[node_id];
                if (neighbors.size() < max_degree) continue;

                sort_neighbors(neighbors);

                // select appropriate edge
                vector<bool> added(n);
                vector<Neighbor> new_neighbors;
                new_neighbors.emplace_back(neighbors.front());

                for (const auto& candidate : neighbors) {
                    const auto candidate_data = dataset.find(candidate.id);

                    bool good = true;
                    for (const auto& new_neighbor : new_neighbors) {
                        const auto new_neighbor_data =
                                dataset.find(new_neighbor.id);
                        const auto dist = calc_dist(
                                candidate_data, new_neighbor_data);

                        if (dist < candidate.dist) {
                            good = false;
                            break;
                        }
                    }

                    if (!good) continue;
                    added[candidate.id] = true;
                    new_neighbors.emplace_back(candidate);

                    if (new_neighbors.size() >= max_degree) break;
                }

                for (const auto& candidate : neighbors) {
                    if (new_neighbors.size() >= max_degree) break;

                    if (added[candidate.id]) continue;
                    added[candidate.id] = true;
                    new_neighbors.emplace_back(candidate);
                }

                edgeset[node_id] = new_neighbors;
            }
        }

        auto save_csv(const string& save_path) {
            ofstream ofs(save_path);
            string line;
            for (int head_id = 0; head_id < n; ++head_id) {
                for (const auto& neighbor_pair : edgeset[head_id]) {
                    line = to_string(head_id) + ',' +
                           to_string(neighbor_pair.id) + ',' +
                           to_string(neighbor_pair.dist);
                    ofs << line << endl;
                }
            }
        }

        auto save_binary(const string& save_path) {
            ofstream ofs(save_path, ios::binary);
            for (int head_id = 0; head_id < n; ++head_id) {
                // line: <K> <id_1> <id_2> ... <id_K>
                const auto neighbors = edgeset[head_id];
                vector<int> line{(int)neighbors.size()};
                for (const auto& neighbor_pair : neighbors) {
                    line.emplace_back(neighbor_pair.id);
                }
                ofs.write((char*)&line[0],
                          (neighbors.size() + 1) * sizeof(int));
            }
        }

        auto save_dir(const string& save_path) {
            vector<string> lines((ceil(n / 1000.0)));
            for (int head_id = 0; head_id < n; ++head_id) {
                const size_t line_i = head_id / 1000;
                for (const auto& neighbor_pair : edgeset[head_id]) {
                    lines[line_i] += to_string(head_id) + "," +
                                     to_string(neighbor_pair.id) + "," +
                                     to_string(neighbor_pair.dist) + "\n";
                }
            }

            for (int i = 0; i < lines.size(); i++) {
                const string path = save_path + "/" + to_string(i) + ".csv";
                ofstream ofs(path);
                ofs << lines[i];
            }
        }

        void save(const string& save_path) {
            // csv
            if (is_csv(save_path))
                save_csv(save_path);
            else if (ends_with(".ivecs", save_path))
                save_binary(save_path);
            else if (ends_with("/", save_path))
                save_dir(save_path);
            else
                throw runtime_error("invalid file type");
        }

        auto load_csv(const string& data_path, const string& graph_path,
                      int degree) {
            dataset.load(data_path);

            ifstream ifs(graph_path);
            if (!ifs) {
                const string message = "Can't open file!: " + graph_path;
                throw runtime_error(message);
            }

            string line;
            while (getline(ifs, line)) {
                const auto row = split<float>(line);
                auto& neighbors = edgeset[row[0]];

                if (neighbors.size() >= degree)
                    continue;

                neighbors.emplace_back(row[2], row[1]);
            }
        }

        auto load_binary(const string& data_path, const string& graph_path,
                         int degree) {
            // read
            dataset.load(data_path);
            ifstream ifs(graph_path, ios::binary);
            vector<vector<int>> lines;
            while (!ifs.eof()) {
                int head;  // number of element
                ifs.read((char*)&head, sizeof(int));

                vector<int> line(head);
                ifs.read((char*)&line[0], head * sizeof(int));

                line.resize(min(head, degree));
                lines.emplace_back(line);
            }

            // make edgeset
            for (int head_id = 0; head_id < n; ++head_id) {
                const auto& line = lines[head_id];
                for (const auto tail_id : line) {
                    const auto dist = calc_dist(
                            dataset.find(head_id), dataset.find(tail_id));
                    edgeset[head_id].emplace_back(dist, tail_id);
                }
            }
        }

        auto load(const string& data_path, const string& graph_path,
                  int degree) {
            if (is_csv(graph_path))
                load_csv(data_path, graph_path, degree);
            else if (ends_with(".ivecs", graph_path))
                load_binary(data_path, graph_path, degree);
            else
                throw runtime_error("invalid file type");
        }

        auto build(const string& data_path, const string& aknng_path) {
            load(data_path, aknng_path, init_degree);
            make_bidirectional();
            optimize_edge();
        }

    };
}

#endif //LGTM_GRAPH_HPP
