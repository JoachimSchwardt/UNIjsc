def fibonacci(n):
    def fib_pair(k):
        if k <= 1:
            return (k,1)
        fh, fh1 = fib_pair(k>>1)
        fk = fh * ((fh1<<1) - fh)
        fk1 = fh1**2+fh**2
        if k & 1:
            fk, fk1 = fk1, fk+fk1
        print(k)
        return (fk, fk1)
    return fib_pair(n)[0]
res=fibonacci(9000000)
