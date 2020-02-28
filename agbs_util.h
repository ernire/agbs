//
// Created by Ernir Erlingsson on 28.2.2020.
//

#ifndef AGBS_AGBS_UTIL_H
#define AGBS_AGBS_UTIL_H

typedef unsigned int uint;
template <class T>
using s_vec = std::vector<T>;
template <class T>
using d_vec = std::vector<std::vector<T>>;

template<typename T>
using random_distribution = std::conditional_t<std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::conditional_t<std::is_floating_point<T>::value,
                std::uniform_real_distribution<T>,
                void>>;

class agbs_util {
public:
    static uint get_block_size(const uint block_index, const uint number_of_samples,
            const uint number_of_blocks) noexcept {
        uint block = (number_of_samples / number_of_blocks);
        uint reserve = number_of_samples % number_of_blocks;
        // Some processes will need one more sample if the data size does not fit completely with the number of processes
        if (reserve > 0 && block_index < reserve) {
            return block + 1;
        }
        return block;
    }

    static uint get_block_start_offset(const uint part_index, const uint number_of_samples,
            const uint number_of_blocks) noexcept {
        int offset = 0;
        for (int i = 0; i < part_index; i++) {
            offset += get_block_size(i, number_of_samples, number_of_blocks);
        }
        return offset;
    }

    template <class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    static void random_uniform(s_vec<T> &vec, const uint vec_size, const uint rand_size) noexcept {
        std::default_random_engine generator(std::random_device{}());
        random_distribution<T> rnd_dist(0, rand_size);
        auto rnd_gen = std::bind(rnd_dist, generator);

        vec.resize(vec_size);
        for (int i = 0; i < vec.size(); ++i) {
            vec[i] = rnd_gen();
        }
    }

    template<class T>
    static T sum_array(T *arr, uint size) noexcept {
        T sum = 0;
        for (uint i = 0; i < size; ++i) {
            sum += arr[i];
        }
        return sum;
    }

    template<class T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
    static void print_array(const std::string &name, T *arr, uint n_dim) noexcept {
        std::cout << name << ": ";
        for (int i = 0; i < n_dim; ++i) {
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    static std::string thousandify(int val) noexcept {
        auto s = std::to_string(val);
        std::string s_new;
//    s_new.resize(s.size() + (s.size()/3));
        std::reverse(s.begin(), s.end());
        for (uint i = 0, j = 0; i < s.size(); ++i) {
            if (i > 0 && i % 3 == 0) {
                s_new.push_back('.');
            }
            s_new.push_back(s[i]);
        }
        std::reverse(s_new.begin(), s_new.end());
        return s_new;
    }
private:

};
#endif //AGBS_AGBS_UTIL_H
