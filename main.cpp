#include <iostream>
#include <random>
#include <functional>
#include <omp.h>
#include <chrono>
#include <cassert>
#include "agbs_util.h"

void measure_duration(const std::string &name, const bool is_out, const std::function<void()> &callback) noexcept {
    auto start_timestamp = std::chrono::high_resolution_clock::now();
    std::cout << name << std::flush;
    callback();
    auto end_timestamp = std::chrono::high_resolution_clock::now();
    if (is_out) {
        std::cout
                << std::chrono::duration_cast<std::chrono::milliseconds>(end_timestamp - start_timestamp).count()
                << " milliseconds\n";
    }
}

template <class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
void sort(s_vec<T> &v_data, s_vec<T> &v_sorted, const uint n_parts) noexcept {

    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    std::cout << "min limit: " << min << ", max limit: " << max << std::endl;
//    T max = 0, min = max(T);
    #pragma omp parallel for schedule(static) reduction(min: min) reduction(max: max)
    for (uint i = 0; i < v_data.size(); ++i) {
        if (v_data[i] < min) {
            min = v_data[i];
        }
        if (v_data[i] > max) {
            max = v_data[i];
        }
    }
    std::cout << "min: " << min << ", max: " << max << std::endl;
    uint n_buckets = n_parts * log(max - min);
    std::cout << "buckets: " << n_buckets << std::endl;
    uint bucket_cnts[n_buckets];
    float divisor = (float)(max - min) / n_buckets;
    std::fill(&bucket_cnts[0], &bucket_cnts[0] + n_buckets, 0);
    // optional cache
    s_vec<uint> v_bucket_index(v_data.size());
    #pragma omp parallel for schedule(static) reduction(+:bucket_cnts[0:n_buckets])
    for (uint i = 0; i < v_data.size(); ++i) {
        uint index = (uint)((v_data[i] - min) / divisor);
        if (index == n_buckets) {
            --index;
        }
        ++bucket_cnts[index];
        v_bucket_index[i] = index;
    }
//    print_array("bucket cardinality: ", &bucket_cnts[0], n_buckets);
    uint sum = agbs_util::sum_array(&bucket_cnts[0], n_buckets);
//    assert(sum == v_data.size());
    // Deterministic assignment
    uint part_sum = 0;
    uint last_part_sum = 0;
    float optimal_sum = (float)v_data.size() / n_parts;
    float score = MAXFLOAT;
    s_vec<uint> v_part_bound;
    v_part_bound.reserve(n_parts+1);
    v_part_bound.push_back(0);
    float tmp;
    // TODO in parallel ?
    for (int i = 0; i < n_buckets && v_part_bound.size() < n_parts; ++i) {
        last_part_sum = part_sum;
        part_sum += bucket_cnts[i];
        tmp = part_sum / optimal_sum;
//        std::cout << "part sum: " << part_sum << " score: " << score << " tmp: " << tmp << std::endl;
        if (std::abs(1.0f-tmp) < score) {
            score = std::abs(1.0f-tmp);
        } else {
            v_part_bound.push_back(i);
            score = MAXFLOAT;
            float change = optimal_sum - last_part_sum;
            optimal_sum += change;
            part_sum = bucket_cnts[i];
        }
    }
    v_part_bound.push_back(n_buckets);
    std::cout << "optimal sum: " << optimal_sum << std::endl;
    agbs_util::print_array("part index bounds: ", &v_part_bound[0], v_part_bound.size());

    uint sums[n_parts];
    std::fill(&sums[0], &sums[0] + n_parts, 0);
    for (uint i = 0; i < v_part_bound.size()-1; ++i) {
        uint begin = v_part_bound[i];
        uint end = v_part_bound[i+1];
        for (uint j = begin; j < end; ++j) {
            sums[i] += bucket_cnts[j];
        }
    }
    agbs_util::print_array("part sums: ", &sums[0], n_parts);
    v_sorted.resize(v_data.size());
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint begin = v_part_bound[tid];
        uint end = v_part_bound[tid+1];
        uint offset = 0;
        for (int t = 0; t < tid; ++t) {
            offset += sums[t];
        }
        uint size = sums[tid];
        uint index = 0;
        for (uint i = 0; i < v_bucket_index.size(); ++i) {
            if (v_bucket_index[i] >= begin && v_bucket_index[i] < end) {
                v_sorted[offset+index++] = v_data[i];
            }
        }
//        #pragma omp critical
//        std::cout << "tid: " << tid << " offset: " << offset << " size: " << size << std::endl;
        std::sort(std::next(v_sorted.begin(), offset), std::next(v_sorted.begin(), offset + size));
    }
    std::cout << "Sample and Copy Sort: ";
}

int main() {

    s_vec<int> v_data;
    uint n = 10000000;
    agbs_util::random_uniform(v_data, n, n / 16);
    uint n_threads = 4;
    omp_set_num_threads(n_threads);

//    std::cout << "Thousandify: " << thousandify(1000) << std::endl;

    std::cout << "Sorting with n: " << agbs_util::thousandify(n) << " and t: " << n_threads << std::endl;
    std::cout << "Best set cardinality (per thread): " << agbs_util::thousandify(n/n_threads) << std::endl;

    s_vec<int> v_data_copy = v_data;
    measure_duration("Standard Sort: ", true, [&]() -> void {
        std::sort(v_data_copy.begin(), v_data_copy.end());
    });
    v_data_copy = v_data;
    measure_duration("Optimal Sort: ", true, [&]() -> void {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            uint size = agbs_util::get_block_size(tid, v_data.size(), n_threads);
            uint offset = agbs_util::get_block_start_offset(tid, v_data.size(), n_threads);
            std::sort(std::next(v_data_copy.begin(), offset), std::next(v_data_copy.begin(), offset+size));
        }
    });
    s_vec<int> v_sorted;
    measure_duration("", true, [&]() -> void {
        sort(v_data, v_sorted, n_threads);
    });
    assert(v_data.size() == v_sorted.size());
    for (uint i = 1; i < v_sorted.size(); ++i) {
        if (v_sorted[i-1] > v_sorted[i])
            std::cerr << "error: " << i << " : " << v_sorted[i-1] << ", " << v_sorted[i] << std::endl;
        assert(v_sorted[i-1] <= v_sorted[i]);
    }
    std::cout << "DONE!" << std::endl;

    return 0;
}