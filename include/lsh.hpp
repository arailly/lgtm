//
// Created by Yusuke Arai on 2021/11/11.
//

#ifndef LGTM_LSH_HPP
#define LGTM_LSH_HPP

#include <random>
#include <cpputil.hpp>

using namespace std;
using namespace cpputil;

namespace lsh {
    struct VectorHash {
        size_t operator () (const vector<int>& key) const {
            string str;
            for (const auto e : key) str += to_string(e) + ",";
            return hash<string>()(str);
        }
    };

    using HashFunc = function<int(DataArray::Data)>;
    using HashFamilyFunc = function<vector<int>(DataArray::Data)>;
    using HashTable = unordered_map<vector<int>, vector<int>, VectorHash>;

    struct SearchResult {
        time_t time = 0;
        time_t lsh_time = 0;
        time_t graph_time = 0;
        vector<int> result;
        unsigned long n_bucket_content = 0;
        unsigned long n_node_access = 0;
        unsigned long n_distinct_node_access = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        void push_back(const SearchResult& result) {
            results.push_back(result);
        }

        void save(const string& log_path, const string& result_path, int k) {
            ofstream log_ofs(log_path);
            string line = "time,n_bucket_content";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                line = to_string(result.time) + "," +
                       to_string(result.n_bucket_content);
                log_ofs << line << endl;

                for (const auto& data_id : result.result) {
                    line = to_string(query_id) + "," + to_string(data_id);
                    result_ofs << line << endl;
                }

                const auto n_miss = k - result.result.size();
                for (int i = 0; i < n_miss; i++) {
                    line = to_string(query_id) + "," + to_string(-1);
                    result_ofs << line << endl;
                }

                query_id++;
            }
        }
    };

    struct LSHIndex {
        const int m, L;
        const double w;
        int n, dim;
        DataArray dataset;
        vector<HashFamilyFunc> G;
        vector<HashTable> hash_tables;
        mt19937 engine;

        LSHIndex(int n, int dim, int n_hash_func_, double w, int L) :
                dim(dim), n(n), dataset(n, dim),
                m(n_hash_func_), w(w), L(L),
                hash_tables(vector<unordered_map<vector<int>,
                        vector<int>, VectorHash>>(L)),
                engine(42) {}

        auto create_hash_func() {
            cauchy_distribution<double> cauchy_dist(0, 1);
            normal_distribution<double> norm_dist(0, 1);
            uniform_real_distribution<double> unif_dist(0, w);

            vector<double> a;
            for (int i = 0; i < dim; ++i) {
                a.push_back(norm_dist(engine));
            }

            const auto b = unif_dist(engine);

            return [=](DataArray::Data data) {
                const auto ip = inner_product(
                        a.begin(), a.end(), data, 0.0);
                return static_cast<int>((ip + b) / (w * 1.0));
            };
        }

        auto create_hash_family() {
            vector<HashFunc> hash_funcs;
            for (int i = 0; i < m; i++) {
                const auto h = create_hash_func();
                hash_funcs.push_back(h);
            }

            return [=](DataArray::Data data) {
                vector<int> hash_vector;
                for (const auto& h : hash_funcs)
                    hash_vector.push_back(h(data));
                return hash_vector;
            };
        }

        auto normalize(const Data<>& data) const {
            auto normalized = vector<float>(data.size(), 0);
            const auto origin = Data<>(data.id, vector<float>(data.size(), 0));
            const float norm = euclidean_distance(data, origin);
            for (int i = 0; i < data.size(); i++) {
                normalized[i] = data[i] / norm;
            }
            return Data<>(data.id, normalized);
        }

        void insert(DataArray::Data data, int id) {
#pragma omp parallel for num_threads(L) schedule(dynamic, 1)
            for (int i = 0; i < L; i++) {
                const auto key = G[i](data);
                auto& hash_table = hash_tables[i];
                auto& val = hash_table[key];
                val.emplace_back(id);
            }
        }

        void build(const DataArray& in_dataset) {
            // set hash function
            dataset = in_dataset;
            for (int i = 0; i < L; i++)
                G.push_back(create_hash_family());

            // insert dataset into hash table
            for (int id = 0; id < n; ++id)
                insert(dataset.find(id), id);
        }

        void build(const string& data_path) {
            for (int i = 0; i < L; i++)
                G.push_back(create_hash_family());

            // insert dataset into hash table
            dataset.load(data_path);
            for (int id = 0; id < n; ++id)
                insert(dataset.find(id), id);
        }

        auto find(DataArray::Data query, int limit = -1) {
            vector<int> result;
            bool is_enough = false;

            for (int i = 0; i < L; i++) {
                HashTable& hash_table = hash_tables[i];
                const auto key = G[i](query);
                for (const auto& data_id : hash_table[key]) {
                    result.emplace_back(data_id);
                    if (limit != -1 && result.size() >= limit) {
                        is_enough = true;
                        break;
                    }
                }
                if (is_enough) break;
            }
            return result;
        }

        auto find_table(DataArray::Data query, int table_id) {
            HashTable& hash_table = hash_tables[table_id];
            const auto key = G[table_id](query);
            return hash_table[key];
        }
    };
}

#endif //LGTM_LSH_HPP
