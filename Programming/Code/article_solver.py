# -*- coding: utf-8 -*-
"""
High performance numerical solution to the generalized article problem using
the 'numba JIT-compiler' in 'no-python' mode (njit).

Prime number factorization is performed with the number-theory module of sympy.
"""

from numba import njit
import numpy as np
from sympy.ntheory import factorint
import time

def _factorint(n, dim):
    """
    The function 'sympy.ntheory.factorint' returns a dictionary, where the
    keys correspond to prime factors and their value to the number of times,
    the factor appears in the number 'n'. For our use, we also have to add 
    '2 * (D - 1)' time a factor of '2' and '5'. for cases, where these do not
    appear in 'n', a 'try-except'-structure is used.
    The result is returned as two 'numpy-Arrays' containing the prime factors
    in 'primes' and their number of appearances in 'powers'.
    """
    factor_dict = factorint(n)      # prime factors of 'n'
    const = 2 * (dim - 1)           # constant for adding factors of '2', '5'
    for key in [2, 5]:
        try:
            factor_dict[key] += const
        except KeyError:
            factor_dict[key] = const
            
    primes, powers = np.array(list(factor_dict.items())).T
    return primes, powers

@njit
def _initial_powers(powers, dim):
    """
    Returns a 'dim * len(powers)' - numpy-Array, where the sum of each column
    'k' is given by 'pk = powers[k]'. This 'sum' is then distributed evenly
    among the rows of each column.
    """
    m = np.zeros((dim, powers.shape[0]))
    for k in range(powers.shape[0]):
        min_power = powers[k] // dim           # minimum power for each row
        rest_power = powers[k] % dim           # rest for uneven distribution
        m[:, k] = min_power
        m[:rest_power, k] += 1
    return m

@njit
def _numbers(p, m):
    """return np.prod(p**m, axis=1)"""
    num = np.zeros(m.shape[0])
    for i in range(m.shape[0]):
        num[i] = np.prod(p**m[i])
    return num

# @njit
# def _delta_sub_ind(j, dim, nonzero_ind):
#     l_ind = np.arange(dim)
#     return l_ind[nonzero_ind]

@njit
def _delta(p, m, num, dim):
    """
    Returns a 'len(p) * dim * dim' - numpy-Array, where the elements are given
        by 'Delta[k,j,l] = (p[k] - 1) * (num[j] - num[l] / p[k])'.
    """
    delta = np.full((p.shape[0], dim, dim), np.inf)
    arange_dim = np.arange(dim)
    for k in range(p.shape[0]):
        nonzero_ind = np.where(m[:, k] > 0)[0]    # 'm' must be >= 0
        for j in range(dim):
            # l_ind = _delta_sub_ind(j, dim, nonzero_ind)
            l_ind = arange_dim[nonzero_ind]
            for l_i in range(l_ind.shape[0]):
                l = l_ind[l_i]
                if l == j:    # leave all diagonals as infinity
                    continue
                val = (p[k] - 1) * (num[j] - num[l] // p[k])
                delta[k, j, l] = val
    return delta

@njit
def _solver(m, num, p, cost, dim, max_depth=100):
    """
    Iteratively searches for a solution to the generalized article problem.
    Step-by-step construction:
        1. Initialize a look-up-table, to avoid repetitions during the loop.
        2. Calculate all possible changes to the 'sum' after possible steps.
        3. Absolute differences between 'sum' and 'cost' after possible steps.
        4. Find the minimal absolute difference not contained in the 
           look-up-table to make the best possible step, that has not been
           tried before.
        5. Store the found absolute difference in the look-up-table.
        6. Find the indexes of 'delta' corresponding to the absolute
           difference and execute the step.
        7. Update the 'sum', the powers 'm' and the numbers 'num' accordingly.
    """
    delta_min_array = np.full(max_depth + 1, np.inf)
    _sum = np.sum(num)
    delta_min_array[0] = np.abs(_sum - cost) 
    
    _flag = (_sum == cost)
    for depth in range(max_depth):
        delta = _delta(p, m, num, dim)
        
        # calculate absolute differences to 'cost' after possible 'delta'
        abs_diff = np.abs(delta + _sum - cost)
            
        # find best step that has not been tried before
        sorted_abs_diff = np.sort(np.unique(abs_diff))
        for i in range(sorted_abs_diff.shape[0]):
            delta_min = sorted_abs_diff[i]
            if delta_min == 0:
                _flag = True
                
            if delta_min in delta_min_array:
                continue
            else:
                delta_min_array[depth + 1] = delta_min
                ind = np.where(abs_diff == delta_min)
                k, j, l = ind[0][0], ind[1][0], ind[2][0]
                # housekeeping
                _sum += delta[k, j, l]        
                m[j, k] += 1                  
                m[l, k] -= 1
                num = _numbers(p, m)
                break
            
        if _flag:
            break
    return m, num, _sum, _flag, depth, delta_min_array


def solver(cost, dim, max_depth=100):
    """
    Returns a count of 'dim' numbers, whose product and sum are identical.
    The numbers 'num' and their total sum 'cost' are interpreted as cents.
    According to the original 'Four Articles Problem', the product of the
    numbers has to be calculated from their dollar representation, but using
    cents allows transforms everything to integers. Thus, '100 + 100 = 200'
    but '100 * 100 = 100', since that corresponds to '1.00 * 1.00 = 1.00'.
    
    This function is merely a wrapper for the actual '_solver'-method. 
    The maximum search depth can be increased. Note that a value of '_sum'
    much larger than 'cost' after 'max_depth' steps is an empirical indicator,
    that no solution exists. However, there is no proof that the algorithm is
    complete as of March 18, 2021.
    """
    if cost < 1 or type(cost) != int:
        print("Cost must be an integer greater than 0 but was " + 
              f"{cost} of type {type(cost)}!")
        raise Warning 
    if dim < 1 or type(dim) != int:
        print("Dimension 'dim' must be an integer greater than 0 but was " + 
              f"{dim} of type {type(dim)}!")
        raise Warning
    
    
    p, powers = _factorint(cost, dim)
    m = _initial_powers(powers, dim)
    num = _numbers(p, m)
    t1 = time.perf_counter()
    m, num, _sum, _flag, depth, delta_min_array =\
        _solver(m, num, p, cost, dim, max_depth)
    t2 = time.perf_counter()
    output = ""
    if _flag:
        output += f"Solution {num} in "
    else:
        output += "No solution after "
        
    return m, num, _sum, _flag, depth, output, t2-t1, np.min(delta_min_array)

def main(cost, dim):       
    m, num, _sum, _flag, depth, output, dt, delta_min = \
        solver(cost, dim, max_depth=500)
    if _flag:
        print(f"Solution for cost={cost} and dim={dim} in {dt} sec: ", num)
    else:
        print(f"No solution after {depth} steps, best approx: ", 
              num, delta_min)
    return _flag
    

if __name__ == "__main__":
    """7470 solvable for dim = 4...39 (822 steps for 25)"""
    cost, dim = 7470, 3  
    main(cost, dim)
