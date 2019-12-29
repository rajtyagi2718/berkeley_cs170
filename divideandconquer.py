from collections import deque
from math import floor, ceil
from random import randint

#####################
# binary arithmetic #
#####################

def add_binary(x, y):
    """Return binary string x+y given x and y as binary strings."""
    # & and, ^ xor, | or
    # make table of 000 001 ... 111 and compute next digit, next carry
    x, y = sorted(map(lambda z: list(map(int, z)), [x, y]), key=len)
    m, M = len(x), len(y)
    result = deque()
    carry = 0
    for i in range(-1, -m-1, -1):
        z = x[i] ^ y[i] ^ carry
        carry = (x[i] ^ y[i] & carry) | (x[i] & y[i])
        result.appendleft(str(z))
    for i in range(-m-1, -M-1, -1):
        z = y[i] ^ carry
        carry = y[i] & carry
        result.appendleft(str(z))
    if carry:
        result.appendleft(carry)
    return ''.join(result)

def subtract_binary(x, y):
    pass

def multiply_binary(x, y):
    """Return binary string x*y given x and y as binary strings."""
    n = max(map(len, [x, y]))
    if n == 1:
        if x == '1' and y == '1':
            return '1'
        return '0'
    m = ceil(n/2)
    xL, xR = x[:m], x[m:]
    yL, yR = y[:m], y[m:]
    P1 = multiply_binary(xL, yL)
    P2 = multiply_binary(xR, yR)
    P3 = multiply_binary(add_binary(xL, yR), add_binary(xR, yL))
    # return P1+'0'*n  + ((P3 - P1 - P2)+'0'*(n//2) + P2
    pass


###################
## binary search ## # already sorted, logn search
###################

def binary_search(lst, x): # O(logn) where n = len(lst)
    """Given sorted list a, return index of x if it exists, -1 otherwise."""
    start, stop = 0, len(lst)
    while start != stop:
        m = (start + stop)//2
        if lst[m] > x:
            start = m+1
        elif lst[m] < x:
            stop = m
        else:
            return m
    return -1

def binary_search_floor(lst, x): # O(logn) where n = len(lst)
    """Find the greatest item in lst less than or equal to x. Return its index.
       If no item exists, return -1. i.e. x < lst[0] xor lst[i] <= x < lst[i+1] xor lst[-1] <= x."""
    start, stop = 0, len(lst)
    while start < stop-1:
        m = (start + stop)//2
        if x >= lst[m]:
            start = m
        else:
            stop = m
    if start == stop-1 and x >= lst[start]:
        return start
    # if (start == stop-1 and x < lst[start]) or start == stop i.e. empty lst
    return -1
    

def binary_search_ceil(lst, x): # O(logn) where n = len(lst)
    """Find the smallest item in lst greater than or equal to x. Return its index.
       If no item exists, return len(lst). i.e. x <= lst[0] xor lst[i-1] < x <= lst[i] xor lst[-1] < x."""
    start, stop = 0, len(lst)
    while start < stop-1:
        m = (start+stop-1)//2
        # 0 1 2 3  goes to  0 1  xor  2 3
        #   m                 m         m
        if x <= lst[m]:
            stop = m+1
        else:
            start = m+1
    if start == stop-1 and x <= lst[start]:
        return start
    # if (start == stop-1 and x > lst[start]) or start == stop i.e. empty lst
    return len(lst)


###############
## mergesort ## # nlogn left right split, bottum up merge
###############

def mergesort_recursive(lst): # O(nlogn)
    """Sort lst recursively by sorting left and right halves separately, merging result."""
    if len(lst) < 2:
        return lst
    m = len(lst)//2
    left = mergesort_recursive(lst[:m])
    right = mergesort_recursive(lst[m:])
    return _merge(left, right)

def _merge(left, right):
    """Merge two sorted lists into one sorted list."""
    if not left:
        return right
    if not right:
        return left
    if left[-1] <= right[0]:
        return left + right
    if right[-1] <= left[0]:
        return right + left
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    if i == len(left):
        result.extend(right[j:])
    else:
        result.extend(left[i:])
    return result

def mergesort_iterative(lst): # O(nlogn)
    """Sort list iteratively using a queue, mergeing and appending first two lsts."""
    if len(lst) < 2:
        return lst
    queue = deque([list(x) for x in lst])
    while len(queue) > 1:
        queue.append(_merge(queue.popleft(), queue.popleft()))
    return queue.popleft()


###############
## quicksort ## # pivot, left middle right
###############

def quickselect(lst, k): # average O(n)
    """Return nonnegative k-th smallest item from unsorted lst, None if it doesn't exist."""
    start, stop = 0, len(lst)
    if k < start or k >= stop:
        return None
    while True:
        _swap(lst, start, randint(start, stop-1))
        lstop = start+1
        estop = start+1
        gstart = stop
        # start lstart ... lstop/estart ... estop-1 [???] gstart ... stop
        while estop < gstart:
            if lst[estop] < lst[start]:
                _swap(lst, lstop, estop)
                lstop += 1
                estop += 1
            elif lst[estop] > lst[start]:
                gstart -= 1
                _swap(lst, estop, gstart)
            else:
                estop += 1
        lstop -= 1
        _swap(lst, start, lstop)
        if k < lstop:
            stop = lstop
        elif k < estop:
            return lst[k]
        else:
            start = gstart

def _swap(lst, i, j):
    """Swap items indexed i and j in lst."""
    lst[i], lst[j] = lst[j], lst[i]
    
def quicksort(lst):
    """Sort lst by partition by random item."""
    _quicksort_range(lst, 0, len(lst))

def _quicksort_range(lst, start, stop):
    if start == stop:
        return None
    _swap(lst, start, randint(start, stop-1))
    lstop = start+1
    estop = start+1
    gstart = stop
    # start lstart ... lstop/estart ... estop-1 [???] gstart ... stop
    while estop < gstart:
        if lst[estop] < lst[start]:
            _swap(lst, lstop, estop)
            lstop += 1
            estop += 1
        elif lst[estop] > lst[start]:
            gstart -= 1
            _swap(lst, estop, gstart)
        else:
            estop += 1
    lstop -= 1
    _swap(lst, start, lstop)
    _quicksort_range(lst, start, lstop)
    _quicksort_range(lst, gstart, stop)

def FFT:
    pass


def quick_sort(a, start=0, stop=None):
    """Given an integer array, sort a[start:stop] in nlogn time."""
    if not stop:
        stop = len(a)
    # base case
    if stop - start < 2:
        return a
    # swap random pivot to front
    i = random.randint(start, stop-1)
    a[start], a[i] = a[i], a[start]
    # partition
    # start/pivot lstart ... lstop/estart ... estop/gstart ... gstop/stop
    # start less ... less equal ... equal ? ... ? greater ... greater
    s, t = start+1, stop
    i = s
    while i < t:
        if a[start] < a[i]:
            t -= 1
            a[t], a[i] = a[i], a[t]
        elif a[start] > a[i]:
            a[s], a[i] = a[i], a[s]
            s += 1
            i += 1
        else:
            i += 1
    # start less ... less equal/s ... equal greater/i/t ... greater/stop
    s -= 1
    a[start], a[s] = a[s], a[start]
    quick_sort(a, start, s)
    quick_sort(a, t, stop)