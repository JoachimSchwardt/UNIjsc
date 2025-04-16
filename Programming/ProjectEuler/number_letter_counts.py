#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count the number of letters used for numbers up to 1000.
"""

LUT = {0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five', 
       6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten', 
       11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen', 
       15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen', 
       19 : 'nineteen', 20 : 'twenty', 30 : 'thirty', 40 : 'forty', 
       50 : 'fifty', 60 : 'sixty', 70 : 'seventy', 80 : 'eighty', 90 : 'ninety', 
       100 : 'hundred', 
       1000 : 'thousand'}



def get_string(number):
    if number >= 10**4:
        print("Warning, numbers above 10000 not supported!")
        return 1
    
    if number == 0:
        return LUT[0]
    
    array = []
    if number < 0:
        number *= -1
        array.append("minus")
    
    digits = [int(dig) for dig in str(number)]
    digits = [0] * (4 - len(digits)) + digits
    
    if digits[-4]:
        array.append(LUT[digits[0]])
        array.append(LUT[1000])
        
    if digits[-3]:
        array.append(LUT[digits[1]])
        array.append(LUT[100])
        
    if number >= 100 and (digits[-1] or digits[-2]):
        array.append("and")
        
    if digits[-2] == 0:
        if digits[-1]:
            array.append(LUT[digits[-1]])
    elif digits[-2] == 1:
        array.append(LUT[10 + digits[-1]])
    else:
        array.append(LUT[10 * digits[-2]])
        if digits[-1]:
            array.append("-")
            array.append(LUT[digits[-1]])
    
    string = " ".join(array).replace(" - ", "-")
    return string


def count_alpha(number):
    ctr = 0
    for letter in number:
        if letter.isalpha():
            ctr += 1
    return ctr


def main():
    print(__doc__)
    
    n = 1000
    
    count = 0
    for i in range(1, n+1):
        string = get_string(i)
        # print(i, string)
        count += count_alpha(string)
    
    print(f"Solution for {n = } is {count = }")
    return 0


if __name__ == "__main__":
    main()
