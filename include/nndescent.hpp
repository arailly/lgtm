//
// Created by Yusuke Arai on 2021/11/10.
//

#ifndef LGTM_NNDESCENT_HPP
#define LGTM_NNDESCENT_HPP

#include <cpputil.hpp>
#include <random>

using namespace std;
using namespace cpputil;

namespace nndescent {
    struct AKNNG {
        int n, dim, K;
        DataArray dataset;
        vector<multimap<float, int>> edgeset;
        vector<unordered_map<int, bool>> added_list;
        mt19937 engine;

        AKNNG(int n, int dim, int K) :
                n(n), dim(dim), K(K),
                dataset(n, dim), edgeset(n), added_list(n),
                engine(42) {}

        auto calc_dist(DataArray::Data data_1,
                       DataArray::Data data_2) {
            float res = 0;
            for (int i = 0; i < dim; ++i, ++data_1, ++data_2) {
                float tmp = *data_1 - *data_2;
                res += tmp * tmp;
            }
            return sqrt(res);
        }

        auto get_neighbors_list() {
            vector<vector<int>> neighbors_list(n);

            for (int i = 0; i < n; ++i) {
                for (const auto& neighbor : edgeset[i]) {
                    // add neighbor
                    neighbors_list[i].emplace_back(neighbor.second);
                    // add reverse neighbor
                    neighbors_list[neighbor.second].emplace_back(i);
                }
            }

            return neighbors_list;
        }

        auto add_neighbor(int head_id, int tail_id) {
            auto& added = added_list[head_id];
            if (head_id == tail_id || added.find(tail_id) != added.end())
                return 0;

            const auto dist = calc_dist(
                    dataset.find(head_id), dataset.find(tail_id));

            auto& neighbors = edgeset[head_id];

            if (neighbors.size() < K) {
                neighbors.emplace(dist, tail_id);
                added[tail_id] = true;
                return 1;
            }

            const auto furthest_itr = --neighbors.cend();
            const auto [furthest_dist, furthest_id] = *furthest_itr;
            if (dist >= furthest_dist) return 0;

            neighbors.emplace(dist, tail_id);
            added[tail_id] = true;

            if (neighbors.size() > K) {
                neighbors.erase(furthest_itr);
                added.erase(furthest_id);
            }

            return 1;
        }

        void build(const string& data_path) {
            // init dataset
            dataset.load(data_path);

            // init edges
            uniform_int_distribution<int> dist(0, n - 1);
            for (int head_id = 0; head_id < n; ++head_id) {
                auto& neighbors = edgeset[head_id];

                while (neighbors.size() < K) {
                    const auto random_id = dist(engine);
                    add_neighbor(head_id, random_id);
                }
            }

            auto n_itr = 0;
            while (true) {
                long long int n_updated = 0;
                const auto neighbors_list = get_neighbors_list();
#pragma omp parallel
                {
#pragma omp for schedule(dynamic, 1000) nowait reduction(+:n_updated)
                    for (int head_id = 0; head_id < n; ++head_id) {
                        for (const auto neighbor_id_1 : neighbors_list[head_id]) {
                            for (const auto neighbor_id_2 : neighbors_list[neighbor_id_1]) {
                                n_updated += add_neighbor(head_id, neighbor_id_2);
                            }
                        }
                    }
                };
                cout << "iteration: " << n_itr << ", update: " << n_updated << endl;
                if (n_updated <= 0) break;
                ++n_itr;
            }
        }

        auto save_csv(const string& save_path) {
            ofstream ofs(save_path);
            string line;
            for (int head_id = 0; head_id < n; ++head_id) {
                for (const auto& neighbor_pair : edgeset[head_id]) {
                    line = to_string(head_id) + ',' +
                           to_string(neighbor_pair.second) + ',' +
                           to_string(neighbor_pair.first);
                    ofs << line << endl;
                }
            }
        }

        auto save_binary(const string& save_path) {
            ofstream ofs(save_path, ios::binary);
            for (int head_id = 0; head_id < n; ++head_id) {
                // line: <K> <id_1> <id_2> ... <id_K>
                vector<int> line{K};
                for (const auto& neighbor_pair : edgeset[head_id]) {
                    line.emplace_back(neighbor_pair.second);
                }
                ofs.write((char*)&line[0], (K + 1) * sizeof(int));
            }
        }

        auto save_dir(const string& save_path) {
            vector<string> lines(static_cast<unsigned long>(ceil(n / 1000.0)));
            for (int head_id = 0; head_id < n; ++head_id) {
                const size_t line_i = head_id / 1000;
                for (const auto& neighbor_pair : edgeset[head_id]) {
                    lines[line_i] += to_string(head_id) + "," +
                                     to_string(neighbor_pair.second) + "," +
                                     to_string(neighbor_pair.first) + "\n";
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

        auto load_csv(const string& data_path, const string& graph_path) {
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

                if (neighbors.size() >= K)
                    continue;

                neighbors.emplace(row[2], row[1]);
            }
        }

        auto load_binary(const string& data_path, const string& graph_path) {
            dataset.load(data_path);
            ifstream ifs(graph_path, ios::binary);
            vector<vector<int>> lines;
            while (!ifs.eof()) {
                vector<int> line(K + 1);
                ifs.read((char*)&line[0], (K + 1) * sizeof(int));
                lines.emplace_back(line);
            }

            for (int head_id = 0; head_id < n; ++head_id) {
                const auto& line = lines[head_id];

                if (line[0] != K)
                    throw runtime_error("degree not matched");

                for (int i = 1; i < K + 1; ++i) {
                    const auto tail_id = line[i];
                    const auto dist = calc_dist(
                            dataset.find(head_id), dataset.find(tail_id));
                    edgeset[head_id].emplace(dist, tail_id);
                }
            }
        }

        auto load(const string& data_path, const string& graph_path) {
            if (is_csv(graph_path))
                load_csv(data_path, graph_path);
            else if (ends_with(".ivecs", graph_path))
                load_binary(data_path, graph_path);
            else
                throw runtime_error("invalid file type");
        }
    };
}

#endif //LGTM_NNDESCENT_HPP
