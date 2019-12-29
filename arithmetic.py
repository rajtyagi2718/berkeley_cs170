from collections import deque
import random, unittest

def addByDigit(x, y, base=10): # O(n) where n is the number of bits of max(x,y) and base=2
    """Add digit by digit using strings."""
    #                               c1 = (c0 + xm-i-1 + ym-i-1) // 10
    #                xm-1 xm-2 ... xm-i xm-i-1 ... x1 x0
    # +  yM-1 yM ... ym-1 ym-2 ... ym-i ym-i-1 ... y1 y0
    # --------------------------------------------------
    #                                   dm-i-1 = (c0 + xm-i-1 + ym-i-1) % 10
    xLst, yLst = map(lambda z: [int(i) for i in str(z)], [x, y])
    xLst, yLst = sorted([xLst, yLst], key=len)
    m, M = map(len, [xLst, yLst])
    result = deque() # stack
    carry = 0
    for i in range(-1, -m-1, -1):
        z = xLst[i] + yLst[i] + carry
        carry, digit = divmod(z, base)
        result.appendleft(str(digit))
    for i in range(-m-1, -M-1, -1):
        z = yLst[i] + carry
        carry, digit = divmod(z, base)
        result.appendleft(str(digit))
    if carry:
        result.appendleft(str(carry))
    return int(''.join(result))

def multiplyByBinary(x, y):
    """
    Successively divide x by 2 and multiply y by 2.
    Add entries of y column if x column odd.
    """
    # x*y = (2**n*rn + 2**(n-1)*rn-1 + ... + ro)*y
    #     = rn*2**n*y + rn-1*2**(n-1)*y + ... + ro*y
    #          ------        ----------            -
    if not x or not y:
        return 0
    mLst = []
    while x > 1:
        x, m = divmod(x, 2)
        mLst.append(m)
    mLst.append(1)
    result = 0
    for m in mLst:
        if m:
            result += y
        y *= 2
    return result

def get_binary_list(x):
    """Return the binary form of x as a list of digits.
       Number presented as Two's Complement for n-bits.
       
    >>> get_binaray_list(-9)
    [-1, 1, 0, 0, 1]
    """
    leading_bit = x < 0
    result = deque() # stack
    while x > 0,
        x, bit = divmod(x, 2)
        result.appendleft(bit)
    result.appendleft(leading_bit)
    return result

def modular_exp(x, y, N): # O(n**3) where n = max_bits(x, y, N)
    """Return x**y (mod N)."""
    # x**y = x**(2*(y//2 + r)) = (x**(y//2))**2 * x**(2*r)
    if y == 0:
        return 1
    q, r = divmod(y, 2)
    z = modular_exp(x, q, N)
    if r:
        return (x * z**2) % N
    return z**2 % N
    
def Euclid_gcd(a, b):
    """Return greatest common divisor of a and b.  a >= b >= 0"""
    # gcd(a, b) == gcd(b, a % b)
    if b == 0:
        return a
    return Euclid_gcd(b, a % b)

def Euclid_extended(a, b): O(n**3)
    """Return int tuple (x, y, g) such that a*x + b*y == g == gcd(a, b). a >= b >= 0"""
    # b*xr + (a % b)*yr = g  i.e.  b*xr + (a - b*(a//b))*yr = g
    if b == 0:
        return (1, 0, a)
    xr, yr, g = Euclid_extend(b, a % b)
    return (yr, xr - (a//b)*yr, g)

def modular_division(x, y, N): O(n**3)
    """Return z such that x = z*y (mod N). If no such z exists, return -1."""
    # z = x*yinv,   y*yinv + c*N = 1,   a*yinv + b*N = g,   g divides y
    a, b, g = Euclid_extend(y, N)
    if g != 1:
        return -1
    # a*y + b*N = 1
    yinv = a
    return x*yinv

from random import randint

def primality_v2(N, k):
    """Tests for primality of a positive integer. Low error probability."""
    for _ in range(k):
        a = randint(1, N-1):
        x = modular_exp(a, N-1)
        if x != 1:
            return False
    return True

def hash_bucket_constants(e, N):
    """Return hash bucket constants based on expected number items e, total number of items N."""
    # prime n > 2*e, N ~ n**k
    n = 2*e+1
    while not primality(n):
        n += 2
    k = 1
    M = n
    while M <= n:
        M *= n
        k += 1
    N = M
    return (N, n, k)

def hash_function_constants(e, N):
    """Return hash function constants that guarantees low collision probability 1/(2*e)."""
    N, n, k = hash_bucket_constants(e, N)
    a = []
    for _ in range(k):
        a.append(randint.(0, n-1))
    return a, n, k
    

class addTest(unittest.TestCase):
    
    def test(self):
        for _ in range(100):
            x, y = map(lambda x: random.randint(10, 10**9), range(2))
            self.assertEqual(addByDigit(x, y), x+y)


if __name__ == '__main__':
    unittest.main()
