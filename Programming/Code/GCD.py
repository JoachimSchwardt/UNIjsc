# Greatest Common Divisor:
import time 

def modulo_small(a, b):
    """Modulo for small integers."""
    while a >= b:
        a -= b
    return a

def modulo(a, b): 
    """
    Functionality based on:
        (x * y) mod b = ((x mod b) * y) mod b 
        (x + y) mod b = (x + y mod b) mod b 
    Let 'n' be a number and 'x' be a digit. Then define 'nx' := n*10 + x:
        'nx' mod b = (x + ((n mod b) * 10) mod b) mod b
    """
    ans = 0
    a = str(a)
  
    # One by one process all digits of 'num' 
    for i in range(0, len(a)): 
        ans = modulo_small(int(a[i]) + ans * 10, b)
    return ans 

def gcd_euclid(a, b):
    """
    Euclid's algorithm for finding the greatest common divisor of 
    two integers greater than 0
    """
    if a == 0 or b == 0:            # handle zero-arguments
        return 0
    if a == b:                      # handle equal arguments
        return a 
    while 1:                        # euclid algorithm
        if a > b:                   # recognize larger argument (a or b)
            a = modulo(a, b)
            if a == 0:
                return b
        else:
            b = modulo(b, a)
            if b == 0:
                return a
        # a, b = max(a, b), min(a, b)
        # a = modulo(a, b)
        # if a == 0:
        #     return b



# Test des Algorithmus
# print("Wähle 2 Zahlen:")
# z1 = int(input("Zahl 1:"))
# z2 = int(input("Zahl 2:"))

# x = gcd_man(z1, z2)

# print(" Der größte gemeinsame Teiler ist: ", x)


# testcases = [(13, 13, 13), (37, 600, 1), (20, 100, 20), 
#               (624129, 2061517, 18913), (657, 11, 1), (0, 324, 0), 
#               (324,0,0),(50,30,10) ] 
       
# for a, b, solution in testcases:
#     print(solution, gcd_euclid(a, b))
def gcd(x, y):
    while y > 0:
        #x, y = y, x % y
        z = x % y
        x = y
        y = z
        print(x, y)
    return x
            
            
x = 59
y = 76
t2 = time.time()
print(gcd(x, y))
t3 = time.time()