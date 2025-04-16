# -*- coding: utf-8 -*-
"""
ObjectOrientedProgramming in Python3. 
"""

class Vehicle:
    def __init__(self, name, max_speed, mileage):
        self.name = name
        self.max_speed = max_speed
        self.mileage = mileage
        
class Bus(Vehicle):
    pass
        
def main():
    print(__doc__)
    
    School_bus = Bus("Volvo", 180, 12)
        
if __name__ == "__main__":
    main()
        
