"""
This file provides a Fraction class as well as 1D and 2D matrices of those.

Look-up table: https://blog.finxter.com/python-dunder-methods-cheat-sheet/
"""
import numpy as np


def gcd(x, y):
    """Computes the greates common divisor using euclids algorithm"""
    while y:
        x, y = y, x % y
    return x


class Fraction():
    """
    This class implements rational numbers as fractions.

    The arithmetic operations are exact, as they are based on integer operations.
    The only two parameters are the numerator 'num' and denominator 'den'.
    Fractions are always reduced on creation using the greatest common divisor.
    """

    def __init__(self, num, den=1):
        """Initializes a fraction and reduces using the gcd."""
        self.num = num
        self.den = den

        self.__reduce__()


    # Arithmetic operations
    def __add__(self, other):
        if isinstance(other, self.__class__):
            num = self.num * other.den + self.den * other.num
            den = self.den * other.den
        elif isinstance(other, FractionArray):
            return other.__add__(self)
        else:
            num = self.num + self.den * other
            den = self.den
        return self.__class__(num, den)


    def __radd__(self, other):
        return self.__add__(other)


    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            self.num = self.num * other.den + self.den * other.num
            self.den *= other.den
        else:
            self.num += self.den * other


    def __sub__(self, other):
        if isinstance(other, self.__class__):
            num = self.num * other.den - self.den * other.num
            den = self.den * other.den
        elif isinstance(other, FractionArray):
            return other.__sub__(self)
        else:
            num = self.num - self.den * other
            den = self.den
        return self.__class__(num, den)


    def __rsub__(self, other):
        return -self.__sub__(other)


    def __isub__(self, other):
        if isinstance(other, self.__class__):
            self.num = self.num * other.den - self.den * other.num
            self.den = self.den * other.den
        else:
            self.num -= self.den * other


    def __mul__(self, other):
        if isinstance(other, self.__class__):
            num = self.num * other.num
            den = self.den * other.den
        elif isinstance(other, FractionArray):
            return other.__mul__(self)
        else:
            num = self.num * other
            den = self.den
        return self.__class__(num, den)


    def __rmul__(self, other):
        return self.__mul__(other)


    def __imul__(self, other):
        if isinstance(other, self.__class__):
            self.num *= other.num
            self.den *= other.den
        else:
            self.num *= other


    def __truediv__(self, other):
        if other == 0:
            print("RuntimeWarning, Fraction divison by zero!")
            print("Returning zero as a fraction...")
            return self.__class__(0)

        if isinstance(other, self.__class__):
            num = self.num * other.den
            den = self.den * other.num
        elif isinstance(other, FractionArray):
            return other.__truediv__(self)
        else:
            num = self.num
            den = self.den * other
        return self.__class__(num, den)


    def __itruediv__(self, other):
        if isinstance(other, self.__class__):
            self.num *= other.den
            self.den *= other.num
        else:
            self.num = self.num
            self.den = self.den * other


    def __pow__(self, other):
        if not isinstance(other, int):
            msg = "Only integers may be used as powers to ensure exact math."
            raise ValueError(msg)

        if self.num == 0:
            if other == 0:      # 0**0 := 1
                return self.__class__(1)
            return self

        if other < 0:
            num = self.den ** (-other)
            den = self.num ** (-other)
        else:
            num = self.num ** other
            den = self.den ** other
        return self.__class__(num, den)


    def __reduce__(self):
        gcd_val = gcd(self.num, self.den)
        self.num //= gcd_val
        self.den //= gcd_val
        return self


    # Comparison operations
    def __eq__(self, other):
        self.__reduce__()
        if isinstance(other, self.__class__):
            return self.num * other.den == self.den * other.num
        if self.den == 1:
            return self.num == other
        if self.den == -1:
            return -self.num == other
        return False


    # Casting operations
    def __neg__(self):
        return self.__class__(-self.num, self.den)


    def __float__(self):
        return self.num / self.den


    def __str__(self, Reduce=True):
        num, den = self.num, self.den
        if Reduce:
            gcd_val = gcd(num, den)
            num //= gcd_val
            den //= gcd_val

        if abs(den) == 1:
            string = fr"{abs(num)}"
        else:
            string = fr"\frac{{{abs(num)}}}{{{abs(den)}}}"

        if num * den < 0:
            string = "-" + string
        return string


    def __repr__(self):
        return self.__str__(Reduce=False)



class FractionArray():
    """
    This class implements arrays of 'Fraction' objects.

    All math operations are based on the 'Fraction'-subroutines.
    All array operations are based on numpy, no explicit loops are needed.
    """

    def __init__(self, num, den=None):
        """
        Initialize a 1D or 2D array of Fraction objects

        Input: 'num' and 'den' must be lists or array of identical shapes
            containing the numerators and denominators
        """
        if isinstance(num, list):
            num = np.array(num)
        if isinstance(den, list):
            den = np.array(den)

        self.ndim = num.ndim
        self.shape = num.shape

        # if no denominators are given assume 'num' is an array of Fractions
        if isinstance(den, type(None)):
            self.arr = num
        elif self.ndim == 1:
            self.arr = np.array([Fraction(num[col], den[col])
                                 for col in range(num.shape[0])])
        elif self.ndim == 2:
            self.arr = np.array([[Fraction(num[row, col], den[row, col])
                                  for col in range(num.shape[1])]
                                 for row in range(num.shape[0])])
        else:
            msg = f"Warning, Fraction arrays of dim = {self.ndim} >= 3 are not supported!"
            raise IndexError(msg)


    def __getitem__(self, index):
        return self.arr[index]


    # Arithemtic operations
    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        return self.__class__(self.arr + other)


    def __radd__(self, other):
        return self.__add__(other)


    def __iadd__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        self.arr += other


    def __sub__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        return self.__class__(self.arr - other)


    def __rsub__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        return self.__class__(other - self.arr)


    def __isub__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        self.arr -= other


    def __mul__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        return self.__class__(self.arr * other)


    def __imul__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        self.arr *= other


    def __rmul__(self, other):
        return self.__mul__(other)


    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        return self.__class__(self.arr / other)


    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            other = other.arr
        res = self.arr @ other

        if isinstance(res, Fraction):
            return res
        return self.__class__(res)


    def __rmatmul__(self, other):
        """Called if 'other' does not have a __matmul__ method."""
        # FIX: Doing 'array @ FractionArray' gives an error despite this,
        #        because arrays supprt __matmul__ (but not for this class)
        print("Call to __rmatmul__ with ", other, self)
        res = other @ self.arr

        if isinstance(res, Fraction):
            return res
        return self.__class__(res)


    def __str__(self):
        string = r'\begin{pmatrix}'
        if self.ndim == 1:
            string = r" \\ ".join([elem.__str__() for elem in self.arr])
        elif self.ndim == 2:
            string = " \\\\\n".join([" & ".join([elem.__str__() for elem in row])
                                     for row in self.arr])
        return '\\begin{pmatrix}\n' + string + '\n\\end{pmatrix}'


    def __repr__(self):
        return self.arr.__repr__().replace("object", "Fraction")


def main():
    """Main function: Test a few 'FractionArray'-operations."""
    arr = FractionArray([5, 7], [-3, 4])
    arr2 = FractionArray([[5, 7], [-3, 4]], [[2, 3], [-6, 8]])
    arr3 = arr2 @ arr
    x = np.arange(2)
    frac = Fraction(5, 2)
    print(7 - frac)
    print(arr3 @ x)


if __name__ == '__main__':
    main()
