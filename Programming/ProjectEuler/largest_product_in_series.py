#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 23:56:34 2022

@author: joachim
"""


# def max_adjacent_product(number: str, n_adj=4):
#     prod = np.prod([int(num) for num in number[:n_adj]])
#     max_prod = prod
#     max_ind = 0
#     ind = n_adj

#     while ind < len(number):
#         num = int(number[ind])
#         if num == 0:
#             ind += n_adj

#         # remove first, add last
#         prod = (prod // int(number[i - n_adj])) * int(number[i])
#         if prod > max_prod:
#             max_prod = prod
#             max_ind = ind

#     return max_prod, max_ind

def max_adjacent_product(number: str, n_adj=4):
    digits = [int(num) for num in number]

    max_prod = 0
    max_ind = 0
    ind = 0
    prod = 0
    ctr = 0
    while ind < len(digits):
        # print(f"{prod = }, {ind = }, {digits[ind] = }, {ctr = }")
        if prod == 0:
            # print("Product was zero, reset counter ... ")
            ctr = 1
            prod = digits[ind]
        elif ctr < n_adj:
            # print("Nonzero product, less than n_adj elements")
            ctr += 1
            prod *= digits[ind]
        else:
            # print("n_adj elements, remove first and add last")
            prod = (prod // digits[ind - n_adj]) * digits[ind]

        if prod > max_prod:
            max_prod = prod
            max_ind = ind - ctr + 1

        ind += 1

    return max_prod, max_ind


def main():
    print(__doc__)

    big_number = """73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
""".replace('\n', '')

    n_adj = 13

    max_prod, max_ind = max_adjacent_product(big_number, n_adj)
    vals = big_number[max_ind:max_ind + n_adj]

    print(f"Solution for {big_number = } is {max_prod = }, {vals = }.")
    return 0


if __name__ == "__main__":
    main()
