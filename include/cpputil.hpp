//
// Created by Yusuke Arai on 2021/11/10.
//

#ifndef LGTM_CPPUTIL_HPP
#define LGTM_CPPUTIL_HPP

#include <iostream>
#include <assert.h>
#include <functional>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <omp.h>
#include <x86intrin.h>

using namespace std;

namespace cpputil {
    template<typename UnaryOperation, typename Iterable>
    Iterable fmap(UnaryOperation op, const Iterable &v) {
        Iterable result;
        std::transform(v.begin(), v.end(), std::back_inserter(result), op);
        return result;
    }

    template<typename Predicate, typename Iterable>
    Iterable filter(Predicate pred, const Iterable &v) {
        Iterable result;
        std::copy_if(v.begin(), v.end(), std::back_inserter(result), pred);
        return result;
    }

    template <typename T = float>
    struct Data {
        size_t id;
        std::vector<T> x;

        Data() : id(0), x({0}) {}

        Data(size_t i, std::vector<T> v) {
            id = i;
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        Data(std::vector<T> v) {
            id = 0;
            std::copy(v.begin(), v.end(), std::back_inserter(x));
        }

        auto& operator [] (size_t i) { return x[i]; }
        const auto& operator [] (size_t i) const { return x[i]; }

        bool operator==(const Data &o) const {
            if (id == o.id) return true;
            return false;
        }

        bool operator!=(const Data &o) const {
            if (id != o.id) return true;
            return false;
        }

        size_t size() const { return x.size(); }
        auto begin() const { return x.begin(); }
        auto end() const { return x.end(); }

        void show() const {
            std::cout << id << ": ";
            for (const auto &xi : x) {
                std::cout << xi << ' ';
            }
            std::cout << std::endl;
        }
    };

    template <typename T = float>
    using Dataset = vector<Data<T>>;

    template <typename T = float>
    using Series = vector<Data<T>>;

    template <typename T = float>
    using RefSeries = vector<reference_wrapper<const Data<T>>>;

    template <typename T = float>
    using SeriesList = vector<vector<Data<T>>>;

    template <typename T = float>
    using DistanceFunction = function<T(Data<T>, Data<T>)>;

    template <typename T = float>
    auto euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
        float result = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            result += std::pow(p1[i] - p2[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

    template <typename T = float>
    auto manhattan_distance(const Data<T>& p1, const Data<T>& p2) {
        float result = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            result += std::abs(p1[i] - p2[i]);
        }
        return result;
    }

    template <typename T = float>
    auto l2_norm(const Data<T>& p) {
        float result = 0;
        for (size_t i = 0; i < p.size(); i++) {
            result += std::pow(p[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }

    template <typename T = float>
    auto clip(const T val, const T min_val, const T max_val) {
        return max(min(val, max_val), min_val);
    }

    template <typename T = float>
    auto cosine_similarity(const Data<T>& p1, const Data<T>& p2) {
        float val = inner_product(p1.begin(), p1.end(), p2.begin(), 0.0)
                    / (l2_norm(p1) * l2_norm(p2));
        return clip(val, static_cast<float>(-1), static_cast<float>(1));
    }

    constexpr float pi = static_cast<const float>(3.14159265358979323846264338);

    template <typename T = float>
    auto angular_distance(const Data<T>& p1, const Data<T>& p2) {
        return acos(cosine_similarity(p1, p2)) / pi;
    }

#ifdef __AVX__
    // function for AVX
    static inline __m128 masked_read(int d, const float *x) {

        assert (0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps (buf);
    }

    float l2_sqr_avx(const float *x, const float *y, size_t d) {

        __m256 msum1 = _mm256_setzero_ps();

        while (d >= 8) {
            __m256 mx = _mm256_loadu_ps (x); x += 8;
            __m256 my = _mm256_loadu_ps (y); y += 8;
            const __m256 a_m_b1 = mx - my;
            msum1 += a_m_b1 * a_m_b1;
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 +=       _mm256_extractf128_ps(msum1, 0);

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps (x); x += 4;
            __m128 my = _mm_loadu_ps (y); y += 4;
            const __m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read (d, x);
            __m128 my = masked_read (d, y);
            __m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
        }

        msum2 = _mm_hadd_ps (msum2, msum2);
        msum2 = _mm_hadd_ps (msum2, msum2);
        return  _mm_cvtss_f32 (msum2);
    }

    auto euclidean_distance_avx(const Data<float>& data1,
                                const Data<float>& data2) {
        const auto dim = data1.size();
        const auto dist = l2_sqr_avx(&data1.x[0], &data2.x[0], dim);
        return dist;
    }
#endif

    auto select_distance(const string& distance = "euclidean") {
        if (distance == "euclidean") {
#ifdef __AVX__
            return euclidean_distance_avx;
#endif
            return euclidean_distance<float>;
        }
        if (distance == "manhattan") return manhattan_distance<float>;
        if (distance == "angular")   return angular_distance<float>;
        else throw runtime_error("invalid distance");
    }

    template <typename T = float>
    vector<T> split(string &input, char delimiter = ',') {
        std::istringstream stream(input);
        std::string field;
        std::vector<T> result;

        while (std::getline(stream, field, delimiter)) {
            result.push_back(std::stod(field));
        }

        return result;
    }

    template <typename T = float>
    Dataset<T> read_csv(const std::string &path, const int& nrows = -1,
                        const bool &skip_header = false) {
        std::ifstream ifs(path);
        if (!ifs) throw runtime_error("Can't open file!");
        std::string line;

        Dataset<T> series;
        for (size_t i = 0; (i < nrows) && std::getline(ifs, line); ++i) {
            // if first line is the header then skip
            if (skip_header && (i == 0)) continue;
            std::vector<T> v = split<T>(line);
            series.push_back(Data<T>(i, v));
        }
        return series;
    }

    const int n_max_threads = omp_get_max_threads();

    template <typename T = float>
    Dataset<T> load_data(const string& path, int n = 0) {
        // file path
        if (path.rfind(".csv", path.size()) < path.size()) {
            auto series = Dataset<T>();
            ifstream ifs(path);
            if (!ifs) throw runtime_error("Can't open file!");
            string line;
            for (size_t i = 0; (i < n) && std::getline(ifs, line); ++i) {
                auto v = split(line);
                series.push_back(Data<T>(i, v));
            }
            return series;
        }

        // dir path
        auto series = Dataset<T>(n * 1000);
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const string data_path = path + '/' + to_string(i) + ".csv";
            ifstream ifs(data_path);
            if (!ifs) throw runtime_error("Can't open file!");
            string line;
            while(getline(ifs, line)) {
                auto v = split(line);
                const auto id = static_cast<size_t>(v[0]);
                v.erase(v.begin());
                series[id] = Data<T>(id, v);
            }
        }
        return series;
    }

    template<typename T>
    void write_csv(const std::vector<T> &v, const std::string &path) {
        std::ofstream ofs(path);
        for (const auto &o : v) {
            std::string line;
            for (const auto &e : o) {
                line += std::to_string(e) + ',';
            }
            line.pop_back();
            line += '\n';
            ofs << line;
        }
    }

    auto get_now() { return chrono::system_clock::now(); }

    auto get_duration(chrono::system_clock::time_point start,
                      chrono::system_clock::time_point end) {
        return chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    bool is_csv(const string& path) {
        return (path.rfind(".csv", path.size()) < path.size());
    }

    constexpr auto double_max = numeric_limits<double>::max();
    constexpr auto double_min = numeric_limits<double>::min();

    constexpr auto float_max = numeric_limits<float>::max();
    constexpr auto float_min = numeric_limits<float>::min();

    struct Neighbor {
        float dist;
        int id;

        Neighbor() : dist(float_max), id(-1) {}
        Neighbor(float dist, int id) : dist(dist), id(id) {}
    };

    using Neighbors = vector<Neighbor>;

    void sort_neighbors(Neighbors& neighbors) {
        sort(neighbors.begin(), neighbors.end(),
             [](const auto& n1, const auto& n2) { return n1.dist < n2.dist; });
    }

    struct CompLess {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist < n2.dist;
        }
    };

    struct CompGreater {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist > n2.dist;
        }
    };

    template <typename T>
    auto scan_knn_search(const Data<T>& query, int k, const Dataset<T>& dataset,
                         string distance = "euclidean") {
        const auto df = select_distance(distance);
        auto threshold = float_max;

        multimap<float, int> result_map;
        for (const auto& data : dataset) {
            const auto dist = df(query, data);

            if (result_map.size() < k || dist < threshold) {
                result_map.emplace(dist, data.id);
                threshold = (--result_map.cend())->first;
                if (result_map.size() > k) result_map.erase(--result_map.cend());
            }
        }

        vector<Neighbor> result;
        for (const auto& result_pair : result_map) {
            result.emplace_back(result_pair.first, result_pair.second);
        }

        return result;
    }

    template <typename T = float>
    auto calc_centroid(const Dataset<T>& dataset) {
        const auto n = dataset.size();
        const auto dim = dataset[0].size();

        // get origin
        Data<T> centroid(vector<T>(dim, 0));

        // calc centroid
        for (const auto& data : dataset) {
            for (int i = 0; i < dim; ++i) {
                centroid[i] += data[i] / n;
            }
        }

        return centroid;
    }

    template <typename T = float>
    auto calc_medoid(const Dataset<T>& dataset) {
        const auto centroid = calc_centroid(dataset);
        const auto search_result = scan_knn_search(centroid, 1, dataset);
        return search_result[0].id;
    }

    auto calc_recall(const Neighbors& actual, const Neighbors& expect) {
        float recall = 0;

        for (const auto& n1 : actual) {
            int match = 0;
            for (const auto& n2 : expect) {
                if (n1.id != n2.id) continue;
                match = 1;
                break;
            }
            recall += match;
        }

        recall /= actual.size();
        return recall;
    }

    auto calc_recall(const Neighbors& actual, const Neighbors& expect, int k) {
        float recall = 0;

        for (int i = 0; i < k; ++i) {
            const auto n1 = actual[i];
            int match = 0;
            for (int j = 0; j < k; ++j) {
                const auto n2 = expect[j];
                if (n1.id != n2.id) continue;
                match = 1;
                break;
            }
            recall += match;
        }

        recall /= actual.size();
        return recall;
    }

    auto load_neighbors(const string& neighbor_path, int n,
                        bool skip_header = false) {
        ifstream ifs(neighbor_path);
        if (!ifs)
            throw runtime_error("Can't open file: " + neighbor_path);

        vector<Neighbors> neighbors_list(n);
        string line;

        if (skip_header) getline(ifs, line);

        while(getline(ifs, line)) {
            const auto row = split(line);

            const int head_id = row[0];
            const int tail_id = row[1];
            const float dist = row[2];

            neighbors_list[head_id].emplace_back(dist, tail_id);
        }

        return neighbors_list;
    }

    struct DataArray {
        vector<float> x;
        int n, dim;

        using Data = vector<float>::const_iterator;

        DataArray(int n, int dim): n(n), dim(dim), x(n * dim) {}

        auto load(const vector<float>& v) { x = v; }

        auto load_fvecs(const string& path) {
            float* row = new float[dim];
            ifstream ifs(path, ios::binary);
            if (!ifs)
                throw runtime_error("can't open file: " + path);

            for (int i = 0; i < n; i++) {
                int head = 0;
                ifs.read((char*)&head, 4);
                ifs.read((char*)row, head * sizeof(float));
                for (int j = 0; j < dim; j++) {
                    x[i * dim + j] = row[j];
                }
            }
        }

        auto load(const string& path) {
            // if path ends with ".fvecs"
            if (path.rfind(".fvecs", path.size()) < path.size())
                load_fvecs(path);
            else
                throw runtime_error("invalid file type");
        }

        decltype(auto) operator[](int i) { return x[i]; }

        decltype(auto) find(int i) {
            return next(x.begin(), i * dim);
        }
    };

    auto euclidean_distance(DataArray::Data data_1, DataArray::Data data_2,
                            int dim) {
        float result = 0;
        for (size_t i = 0; i < dim; i++, ++data_1, ++data_2) {
            result += pow(*data_1 - *data_2, 2);
        }
        result = sqrt(result);
        return result;
    }

    struct GroundTruth {
        int n, k;
        vector<vector<int>> x;

        GroundTruth(int n, int k) : n(n), k(k), x(n) {}

        auto load_ivecs(const string& path) {
            ifstream ifs(path, ios::binary);
            unsigned int *row = new unsigned int[k];
            for (int i = 0; i < n; i++) {
                int head;
                ifs.read((char*)&head, 4);
                ifs.read((char*)row, head * 4);
                for (int j = 0; j < k; ++j) {
                    x[i].emplace_back(row[j]);
                }
            }
        }

        decltype(auto) operator[](int i) { return x[i]; }

        auto load(const string& path) {
            if (path.rfind(".ivecs", path.size()) < path.size())
                load_ivecs(path);
            else
                throw runtime_error("invalid file type");
        }
    };

    auto calc_recall(const Neighbors& actual, const vector<int>& expect,
                     int k) {
        float recall = 0;

        for (int i = 0; i < k; ++i) {
            const auto n1 = actual[i];
            int match = 0;
            for (int j = 0; j < k; ++j) {
                const auto n2 = expect[j];
                if (n1.id != n2) continue;
                match = 1;
                break;
            }
            recall += match;
        }

        recall /= actual.size();
        return recall;
    }

    auto ends_with(const string& pattern, const string& str) {
        return str.rfind(pattern, str.size()) < str.size();
    }

}

#endif //LGTM_CPPUTIL_HPP
