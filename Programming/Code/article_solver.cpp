/** Open "Developer Command Prompt for VS" and compile with "cl /EHsc article_solver.cpp" */

#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <cmath>
#include <unordered_set>
#include <chrono>

using ntype = long long;
using vl = std::vector<long long>;
using vvl = std::vector<vl>;

struct Factorization {
    vl primes;
    vl multiplicities;
};

// class Matrix

inline void _reduce_factor(ntype& number, ntype& divisor, Factorization& factorization, int& indx_prev_factor) noexcept {
    if (number % divisor == 0) {
        factorization.primes.push_back(divisor);
        factorization.multiplicities.push_back(1);
        indx_prev_factor += 1;
        number /= divisor;
        while (number % divisor == 0) {
            factorization.multiplicities[indx_prev_factor] += 1;
            number /= divisor;
        }
    }
}

void _factorize(ntype number, Factorization& factorization){
    int indx_prev_factor = -1;     // assumes factorization is empty!
    for (ntype divisor : {2, 3, 5}) {
        _reduce_factor(number, divisor, factorization, indx_prev_factor);
    }
    static std::array<int, 8> increments = {4, 2, 4, 2, 4, 6, 2, 6};
    int i = 0;
    for (ntype divisor = 7; divisor * divisor <= number; divisor += increments[i++]) {
        _reduce_factor(number, divisor, factorization, indx_prev_factor);
        if (i == 8){
            i = 0;
        }
    }
    if (number > 1){
        factorization.primes.push_back(number);
        factorization.multiplicities.push_back(1);
    }
}

Factorization factorize(ntype number){
    Factorization factorization;
    if (number < 1) {
        factorization.primes.push_back(0);
        factorization.multiplicities.push_back(1);
    } else if (number < 4) {
        factorization.primes.push_back(number);
        factorization.multiplicities.push_back(1);
    }
    else {
        _factorize(number, factorization);
    }
    return factorization;
}

ntype int_power(const ntype number, const ntype power){
    /** https://stackoverflow.com/questions/1505675/power-of-an-integer-in-c */
    if (power == 0) return 1;
    if (power == 1) return number;

    int temp = int_power(number, power / 2);
    if ((power % 2) == 0) return temp * temp; 
    return number * temp * temp; 
}

auto collapse_factorization(const Factorization& factorization){
    auto result = 1;
    for (int indx = 0; indx < factorization.primes.size(); ++indx){
        result *= int_power(factorization.primes[indx], factorization.multiplicities[indx]);
    }
    return result;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& vector){
    out << "[";
    for (int i = 0; i < vector.size() - 1; ++i){
        out << vector[i] << ", ";
    }
    out << vector[vector.size() - 1] << "]";
    return out;
}

std::ostream& operator<<(std::ostream& out, const vvl& matrix){
    out << "[";
    for (int i = 0; i < matrix.size(); ++i){
        if (i > 0){
            out << " ";
        }
        out << "[";
        for (int col = 0; col < matrix[0].size() - 1; ++col){
            out << matrix[i][col] << ", ";
        }
        out << matrix[i][matrix[0].size() - 1] << "]";
        if (i < matrix.size() - 1) {
            out << '\n';
        }
    }
    out << "]";
    return out;
}

std::ostream& operator<<(std::ostream& out, const Factorization& factorization){
    out << "Primes         : " << factorization.primes << '\n' 
        << "Multiplicities : " << factorization.multiplicities;
    return out;    
}


vvl initialize_multiplicity_matrix(const vl& multiplicities, const int dimension){
    /**
    Returns a matrix (vector of vectors) containing 'dimension' rows with the number of columns corresponding to the length of 'multiplicities.
    The sum of each column corresponds to 'multiciplicites[column]'.
    This initializer evenly distributes the factors among the columns the matrix.
    */
    vvl matrix(dimension, vl(multiplicities.size(), 0));
    int indx_row = 0;
    for (int k = 0; k < multiplicities.size(); ++k){
        int min_multiplicity = multiplicities[k] / dimension;
        int remaining_multiplicity = multiplicities[k] % dimension;
        if (min_multiplicity > 0){
            for (int row = 0; row < dimension; ++row){
                matrix[row][k] = min_multiplicity;
            }
        }
        for (int row = indx_row; row < remaining_multiplicity + indx_row; ++row){
            matrix[row % dimension][k] += 1;
        }
        indx_row = (indx_row + remaining_multiplicity) % dimension;
    }
    /* This approach suffers from overflow due to all prime factors being put in the first number */
    // for (int k = 0; k < multiplicities.size(); ++k){
    //     matrix[0][k] = multiplicities[k];
    // }
    /* Even distribution and all remaining factors -> first number; still overflows and massive performance drop too */
    // for (int k = 0; k < multiplicities.size(); ++k){
    //     int min_multiplicity = multiplicities[k] / dimension;
    //     int remaining_multiplicity = multiplicities[k] % dimension;
    //     if (min_multiplicity > 0){
    //         for (int row = 0; row < dimension; ++row){
    //             matrix[row][k] = min_multiplicity;
    //         }
    //     }
    //     matrix[0][k] += remaining_multiplicity;
    // }
    return matrix;
}

vl compute_numbers(const vl& primes, const vvl& matrix){
    vl numbers(matrix.size());
    for (int row = 0; row < matrix.size(); ++row){
        numbers[row] = 1;
        for (int col = 0; col < matrix[0].size(); ++col){
            numbers[row] *= int_power(primes[col], matrix[row][col]);
        }
    }
    return numbers;
}

struct GESP_State{
    /* Generalized Equal-Sum-Product-problem state strcture */
    vvl matrix = {{0}};
    vl numbers = {0};
    ntype sum = 0;
    long depth = 0;
    bool success = false;
};

std::ostream& operator<<(std::ostream& out, const GESP_State& state){
    out << "Matrix  :\n" << state.matrix << '\n'
        << "Numbers :\n" << state.numbers << '\n'
        << "Sum     : " << state.sum << '\n'
        << "Success : " << (state.success ? "True" : "False") << " after " << state.depth << " Iterations" << '\n';
    return out;    
}

void update_factorization(Factorization& factorization, const Factorization& other){
    for (int other_indx = 0; other_indx < other.primes.size(); ++other_indx){
        long other_prime = other.primes[other_indx];
        auto indx = std::find(factorization.primes.begin(), factorization.primes.end(), other_prime);   //TODO: optimize search
        if (indx == factorization.primes.end()){
            factorization.primes.push_back(other_prime);
            factorization.multiplicities.push_back(other.multiplicities[other_indx]);
        } else {
            factorization.multiplicities[std::distance(factorization.primes.begin(), indx)] += other.multiplicities[other_indx];
        }
    }
}

GESP_State solve_gesp(const ntype target_sum, const long dimension, const Factorization& constant_factorization, const int max_depth = 50, const bool verbose = false, const bool silent = false){
    /** 
    Iteratively searches for a solution to the "Generalized Equal-Sum-Product-problem" ::
        Sum(numbers) * Constant == Prodct(numbers) 
    for integer numbers and an arbitrary integer constant (given as prime factorization)

    Step-by-step construction ::
        1. Initialize a look-up-table to avoid repetitions during the loop. We use the total sum as an index to reduce load 
            (but this comes with a large reduction of the available search space over time, potentially dangerous approach that missed qite a few options)
        2. Calculate all possible changes to the 'sum' after possible steps ('discrete jacobian, a.k.a. first derivative in integral calculus)
        3. Absolute differences between 'current_sum' and 'target_sum' after possible steps
        4. Find the minimal absolute difference not contained in the LUT to make the best possible step that has not been tried before
        5. Store the found absolute difference in the LUT (maybe replace by more sophisticated hashing approach)
        6. Find the indices of the 'jacobian' corresponding to the absolute difference and execute the step
        7. Update the 'current_sum', the multiplicity distribution matrix 'matrix' and the 'numbers' accordingly
        8. Iterate until a solution is found or the maximal search depth is reached
    */
    GESP_State state;
    auto factorization = factorize(target_sum);
    update_factorization(factorization, constant_factorization);
    state.matrix = initialize_multiplicity_matrix(factorization.multiplicities, dimension);
    state.numbers = compute_numbers(factorization.primes, state.matrix);
    state.sum = std::accumulate(state.numbers.begin(), state.numbers.end(), 0ll);
    std::unordered_set<ntype> abs_diff_min_set;
    abs_diff_min_set.insert(std::abs(state.sum - target_sum));
    if (state.sum == target_sum){
        state.success = true;
        return state;
    }
    for (int depth = 0; depth < max_depth; ++depth){
        ntype abs_diff_opt = std::numeric_limits<ntype>::max();
        int indx_prime_opt = 0;
        int row_opt = 0;
        int col_opt = 0;
        ntype jacobian_opt = 0;
        for (int indx_prime = 0; indx_prime < factorization.primes.size(); ++indx_prime){
            for (int row = 0; row < dimension; ++row){
                for (int col = 0; col < dimension; ++col){
                    if ((row != col) && (state.matrix[col][indx_prime] != 0)){
                        // jacobian for moving prime_factor at 'indx_prime' from row 'col' to 'row'
                        auto jacobian = (factorization.primes[indx_prime] - 1) * (state.numbers[row] - state.numbers[col] / factorization.primes[indx_prime]);
                        auto abs_diff = std::abs(jacobian + state.sum - target_sum);

                        // check that the new error is smaller than the original one and that the hash is new
                        if ((abs_diff < abs_diff_opt) && (abs_diff_min_set.find(abs_diff) == abs_diff_min_set.end())){
                            abs_diff_opt = abs_diff;
                            indx_prime_opt = indx_prime;
                            row_opt = row;
                            col_opt = col;
                            jacobian_opt = jacobian;
                            // TODO: Implement 2-step look-ahead
                        }
                    }
                }
            }
        }
        if (row_opt == col_opt){
            if (silent == false){
                std::cerr << "No improvement possible after "<< depth << " Iterations for dimension "<< dimension << ", numbers " << state.numbers << ", sum " << state.sum << " and target sum "<< target_sum <<"\n";
            }
            return state;
        }
        abs_diff_min_set.insert(abs_diff_opt);
        ++state.matrix[row_opt][indx_prime_opt];
        --state.matrix[col_opt][indx_prime_opt];
        state.numbers[row_opt] *= factorization.primes[indx_prime_opt];
        state.numbers[col_opt] /= factorization.primes[indx_prime_opt];
        state.sum += jacobian_opt;
        ++state.depth;
        if (verbose){
            std::cerr << state.depth<<", "<< state.numbers<<"\n";
        }
        if (state.sum == target_sum){
            state.success = true;
            return state;
        }
        // std::cout<<"housekeeping: "<<row_opt<<", "<<col_opt<<", "<<indx_prime_opt<<", "<<abs_diff_opt<<'\n';
        // std::cout<<state.numbers<<'\n';
        // if ((depth % 256) == 128){
        //     // find largest number
        //     int indx_max_number = std::distance(state.numbers.begin(), std::max_element(state.numbers.begin(), state.numbers.end()));
        //     std::cout<<indx_max_number<<", "<<state.numbers[indx_max_number]<<", "<<state.numbers<<'\n';
        //     int indx_prime = 0;
        //     for (int indx_min_multiplicity = 0; indx_min_multiplicity < dimension; ++indx_min_multiplicity){
        //         if ((indx_min_multiplicity != indx_max_number) && (state.matrix[indx_min_multiplicity][indx_prime] != 0)){
        //             ++state.matrix[indx_max_number][indx_prime];
        //             --state.matrix[indx_min_multiplicity][indx_prime];
        //             state.numbers[indx_max_number] *= factorization.primes[indx_prime];
        //             state.numbers[indx_min_multiplicity] /= factorization.primes[indx_prime];
        //             state.sum += (factorization.primes[indx_prime] - 1) * (state.numbers[indx_max_number] - state.numbers[indx_min_multiplicity] / factorization.primes[indx_prime]);
        //             break;
        //         }
        //     }
        //     std::cout<<indx_max_number<<", "<<state.numbers[indx_max_number]<<", "<<state.numbers<<'\n';
        // }
    }
    return state;
}

GESP_State solve_gesp_4d(const ntype target_sum, const Factorization& constant_factorization, const bool verbose = false){
    /** Brute-force approach for dimension==4 :: 
    n1+n2+n3+n4 = s  and  n1*n2*n3*n4 = cs
        -> n1+n2+n3+cs/(n1*n2*n3) = s
        -> (s-n1-n2-n3)*n3 = cs/(n1*n2)
        -> n3**2 - n3*(s-n1-n2) + cs/(n1*n2) = 0
        -> n3,n4 = (s-n1-n2)/2 +- sqrt((s-n1-n2)**2/4 - cs/(n1*n2))
    */
    const long long constant = collapse_factorization(constant_factorization);
    const long long target_product = target_sum * constant;
    GESP_State state;
    state.success = false;
    long num1 = 1;
    long num2, num3, num4;
    for (; num1 < target_sum; ++num1){
        num2 = 1;
        for (; num2 < target_sum - num1; ++num2){
            auto root = std::sqrt(((target_sum-num1-num2)*(target_sum-num1-num2)/4 -  constant*target_sum/(num1*num2)));
            num3 = (long) ((target_sum-num1-num2)/2 + root);
            num4 = (long) ((target_sum-num1-num2)/2 - root);
            if ((num1+num2+num3+num4 == target_sum) && (num1*num2*num3*num4 == constant * target_sum)){
                state.numbers = {num1, num2, num3, num4};
                state.success = true;
                if (verbose){
                    std::cerr << num1<<", " << num2<<", " << num3<<", " << num4<<", "<<target_sum<<"\n"<<state.numbers<<"\n";
                }
                return state;
            }
        }
    }
    return state;
}

void write_sgesp_to_file(const std::string& filename, vl& dimensions, vl& target_sums, const long max_depth = 10000){
    /** Compute the special GESP-problem for all given 'dimensions' and 'target_sums' and store the results in a file.
    Header: Constant : 100**(dimension-1),max_depth
            Dimension,Sum,Success,Depth
    Format: dimensions,target_sum,success,depth
    */
    std::ofstream out(filename + ".txt");
    if (out.is_open()){
        out << "c=100**(dimension-1)," << max_depth << '\n';
        out << "Dimension,Sum,Success,Depth\n";
        std::cout << "Constant : 100**(dimension - 1), i.e. special GESP-problem\n"
                  << "Maximal search depth : " << max_depth << "\n"
                  << "Dimensions : " << dimensions[0] << " ... " << dimensions[dimensions.size()-1] << " with stepsize " << dimensions[1] - dimensions[0] << "\n"
                  << "Target sums : " << target_sums[0] << " ... " << target_sums[target_sums.size()-1] << " with stepsize " << target_sums[1] - target_sums[0] << "\n";
        for (const auto dimension : dimensions){
            Factorization constant_factorization = {{2, 5}, {2*(dimension - 1), 2*(dimension - 1)}};
            long counter = 0;
            for (const auto target_sum : target_sums){
                auto state = solve_gesp(target_sum, dimension, constant_factorization, max_depth, false, true);
                out << dimension << "," << target_sum << "," << state.success << "," << state.depth << "\n";
                if (state.success){
                    if (std::accumulate(state.numbers.begin(), state.numbers.end(), 0ll) != target_sum){
                        std::cerr << "CRITICAL ERROR: false-positive success for dimension=" << dimension << ", sum=" << target_sum << "!\n";
                    }
                    ++counter;
                }
            }
            std::cout << "Finished computing " << target_sums.size() << " target sums for dimension " << dimension << " (" << counter << " successful)\n";
        }
        out.close();
        std::cout << "Data was written to " << filename << ".txt\n";
    } else {
        std::cerr << "Error opening file!\n";
    }
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) {
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

int main(){
    /** Diff to Python :: here initialization is more uniform; surprisingly large effect on search depth (sometimes much slower, sometimes much faster)
        -> otherwise algorithm identical!
        Lots of prime factors give stable solutions (e.g.2*3*5*7*11 coherently soluble for dim=3...18 and 2*3*5*7*11*13 for dim=5...292 [1 iteration for dim=226,273,274,282])
     */
    const long dimension = 4;
    const ntype target_sum = 747;
    const Factorization constant_factorization = {{2, 5}, {2*(dimension - 1), 2*(dimension - 1)}};
    const auto constant = collapse_factorization(constant_factorization);
    std::cout << "Dimension : 4,  Constant : " << constant << "\n\n";
    // for (long sum = 600; sum < 1000; ++sum){
    //     auto state = solve_gesp(sum, dimension, constant_factorization, 2000);
    //     // std::cout << "Sum     : " << sum << '\n'
    //     //           << "Numbers : " << state.numbers << " with sum " << state.sum << '\n'
    //     //           << "State   : " << (state.success ? "True" : "False") << " after " << state.depth << " Iterations" << "\n\n";
    //     if (state.success){
    //         std::cout << "Sum : " << sum << ", Numbers : " << state.numbers << " after " << state.depth << " Iterations" << "\n";
    //     }
    //     // auto state2 = solve_gesp_4d(sum, constant_factorization);
    //     // if (state2.success){
    //     //     std::cout << "(2) Sum : " << sum << " with " << state2.numbers << "\n";
    //     // }
    // }
    // for (long dim = 3; dim < 300; ++dim){
    //     const Factorization cfac = {{2, 5}, {2*(dim - 1), 2*(dim - 1)}};
    //     auto state = solve_gesp(2*3*5*7*11*13, dim, cfac, 10000);
    //     if (state.success){
    //         // std::cout << "Dimension : " << dim << ", Numbers : " << state.numbers << " after " << state.depth << " Iterations" << "\n";
    //         std::cout << "Dimension : " << dim << " after " << state.depth << " Iterations" << "\n";
    //     }
    // }
    // std::cout<<solve_gesp_4d(1716, constant_factorization, true)<<'\n';
    // std::cout<<solve_gesp(2*3*5*7*11*13, dimension, constant_factorization, 10000, true)<<'\n';
    // const long dim = 70;
    // const Factorization cfac = {{2, 5}, {2*(dim - 1), 2*(dim - 1)}};
    // std::cout<<solve_gesp(7470, dim, cfac, 10000, false)<<'\n';
    // auto state = solve_gesp(target_sum, dimension, constant_factorization);
    // std::cout<<state<<'\n';
    /* Test simulation */
    auto timer_start = std::chrono::high_resolution_clock::now();
    vl dimensions = arange(2ll, 10ll);
    vl target_sums = arange(1ll, 1000ll);
    write_sgesp_to_file("sgesp_2_10_1_1_1000_1", dimensions, target_sums);
    auto timer_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (timer_end - timer_start);
    std::cout << "Simulation complete in " << duration.count() << " seconds\n";
    std::cout << "Program Terminated successfully!\n";
    return 0;
}