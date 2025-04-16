#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Count the sundays on the first of a month

    1 Jan 1900 was a Monday.
    Thirty days has September,
    April, June and November.
    All the rest have thirty-one,
    Saving February alone,
    Which has twenty-eight, rain or shine.
    And on leap years, twenty-nine.
    A leap year occurs on any year evenly divisible by 4, 
    but not on a century unless it is divisible by 400.

"""

LUT_DAYS = {0 : 'monday', 
            1 : 'tuesday', 
            2 : 'wednesday',
            3 : 'thursday',
            4 : 'friday',
            5 : 'saturday',
            6 : 'sunday',
            }

LUT_DAYS_INV = {val : key for key, val in LUT_DAYS.items()}


LUT_MONTHS = {1 : {'str' : 'january', 'days' : 31},
              2 : {'str' : 'february', 'days' : 28},
              3 : {'str' : 'march', 'days' : 31},
              4 : {'str' : 'april', 'days' : 30},
              5 : {'str' : 'may', 'days' : 31},
              6 : {'str' : 'june', 'days' : 30},
              7 : {'str' : 'july', 'days' : 31},
              8 : {'str' : 'august', 'days' : 31},
              9 : {'str' : 'september', 'days' : 30},
              10 : {'str' : 'october', 'days' : 31},
              11 : {'str' : 'november', 'days' : 30},
              12 : {'str' : 'december', 'days' : 31},
              }


class Date:
    def __init__(self, day, month, year, day_str):
        self.day = day
        self.month = month
        self.year = year
        self.day_str = day_str
        
    def __str__(self):
        string = (f"{self.day} {LUT_MONTHS[self.month]['str']} {self.year} "
                  f"was a {self.day_str}")
        return string
    
    def __repr__(self):
        return self.__str__()
    


def is_leapyear(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True
                
    return leap


def get_next_day(date):
    day = date.day + 1
    month = date.month
    year = date.year
    day_str = LUT_DAYS[(LUT_DAYS_INV[date.day_str] + 1) % 7]
    if day > LUT_MONTHS[date.month]['days']:
        day = 1
        month += 1
    if month > 12:
        month = 1
        year += 1
    
    return Date(day, month, year, day_str)
    
    


def main():
    print(__doc__)
    
    monday = Date(1, 1, 1900, LUT_DAYS[0])
    date = monday
    sunday_ctr = 0
    while date.year < 2001:
        date = get_next_day(date)
        if date.year > 1900:
            if date.day == 1 and date.day_str == LUT_DAYS[6]:
                sunday_ctr += 1
                print(date)
    
    print(f"{sunday_ctr = }")
    return 0

if __name__ == "__main__":
    main()
