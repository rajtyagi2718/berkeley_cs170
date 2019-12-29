



## Exercises ##

def length_longest_increasing_subsequence(a):
    """Return the length of the longest increasing subsequence."""
    if not a:
        return 0
    L = [1]
    for i in range(1, len(a)):
        L.append((1 + max(L[j] for j in range(i) if a[j] < a[i]),
                          default=0))
    return max(L)
    

def longest_incrasing_subsequence(a):
    """Return the longest increasing subsequence."""
    # base cases
    if not a:
        return []
    L = [0]
    prev = [None]
    a = [float('inf')] + a
    
    # induction
    for i in range(1, len(a)):
        j = max(range(i),
                key=lambda j: L[j] if a[j] < a[i] else -float('inf'))
        L.append(1 + L[j])
        prev.append(j)
        
    # build sequence
    i = max(range(len(a)), key=lambda i: L[i])
    result = [None]*L[i]
    for j in range(L[i]-1, -1, -1):
        result[j] = a[i]
        i = prev[i]
    return result


def edit_distance(s, t):
    # S _ N O W Y
    # S U N N _ Y

    # D[i][j] == distance s[:i] to t[:j]
    # base case: i == 0: '' to t[:j]
    D = list(range(len(t)+1))
    E = [[3] + [0]*len(t)]

    edit_types = ['ins', 'del', 'id', 'sub']
    edit_diff = [(0,1), (1,0), (1,1), (1,1)]

    # induction
    for i in range(1, len(s)+1):
        Di = [i]
        Ei = [1]
        for j in range(1, len(t)+1):
            dist = [1+Di[j-1], 1+D[j], int(s[i-1] != t[j-1]) + D[j-1]]
            e = min(range(3), key=lambda e: dist[e])
            Di.append(dist[e])
            if e == 2:
                e += int(s[i-1] != t[j-1])
            Ei.append(e)
        D = Di
        E.append(Ei)

    # construct sequence
    se = []
    te = []
    i = len(s)
    j = len(t)
    while i or j:
        di, dj = edit_diff[E[i][j]]
        se.append(s[i-1] if di else '_')
        te.append(t[j-1] if dj else '_')
        i -= di
        j -= dj
    se = ''.join(x for x in se)[::-1]
    te = ''.join(x for x in te)[::-1]
    return D[-1], se, te


def edit_distance_divide_conquer(s, t):
    # S _ N O W Y
    # S U N N _ Y

    edit_types = ['ins', 'del', 'id', 'sub']
    edit_before = [(0,-1), (-1,0), (-1,-1), (-1,-1)]
    edit_after = [(0, 1), (1, 0), (1,1), (1,1)]

    def prev(D, Di, i, j, s0, t0, direction):
        if not direction:
            diff = int(s[s0+i-1] != t[t0+j-1])
            dist = [1+Di[j-1], 1+D[j], diff + D[j-1]]
        else:
            diff = int(s[s0+i] != t[t0+j])
            dist = [1+Di[j+1], 1+D[j], diff + D[j+1]]
        e = min(range(3), key=lambda e: dist[e])
        d = dist[e]
        if e == 2:
            e += diff
        return d, e

    def forward(s0, s1, t0, t1):
        # D[i][j] == dist from s[s0:s0+i] to t[t0:t0+j]
        D = list(range(t1-t0+1))
        E = [3] + [0]*(t1-t0)
        for i in range(1, s1-s0+1):
            Di = [i]
            Ei = [1]
            for j in range(1, t1-t0+1):
                d, e = prev(D, Di, i, j, s0, t0, 0)
                Di.append(d)
                Ei.append(e)
            D = Di
            E = Ei
        return D, E

    def backward(s0, s1, t0, t1):
        # D[i][j] = dist from s[s1-1: s0-1: -1] adn t[t1-1: t0-1: -1]
        # s1-1 -> s0   t1-1 -> t0
        D = list(range(t1-t0, -1, -1))
        E = [0]*(t1-t0) + [3]
        for i in range(s1-s0-1, -1, -1):
            Di = [None]*(t1-t0) + [D[-1]+1]
            Ei = [None]*(t1-t0) + [0]
            for j in range(t1-t0-1, -1, -1):
                d, e = prev(D, Di, i, j, s0, t0, 1)
                Di[j] = d
                Ei[j] = e
            D = Di
            E = Ei
        return D, E

    def midpoint(s0, s1, t0, t1):
        # optimal path from (s0,t0) to (s1, t1) passes through some (p,q)
        print(s0, s1, t0, t1)
        p = (s0+s1)//2
        Df, Ef = forward(s0, p, t0, t1)
        Db, Eb = backward(p, s1, t0, t1)
        D = [sum(x) for x in zip(Df, Db)]
        q = min(range(t1-t0+1), key=lambda q: D[q])
        mid = [Ef[q]]
        print('   ', mid)
        if s1-s0 == 0:
            return mid
        if s1-s0 == 1:
            return mid
        if s1-s0 == 2:
            return midpoint(s0, p, t0, q) + mid
        return midpoint(s0, p, t0, q) + mid + midpoint(p, s1, q, t1)

    dist = forward(0, len(s), 0, len(t))
    seq = midpoint(0, len(s), 0, len(t))
    return [dist], seq
        
    
    

from bisect import bisect_right
from collections import deque


def knapsack_with_repetition(weights, values, capacity):
    """Items of various weights, values. Choose as many of each item,
       maximize for total value given total weight capacity restriction."""
    # V[i] = max value with capacity i
    perm = {x:i for i,x in enumerate(weights)}
    weights.sort()
    del weights[bisect_right(weights, capacity):]
    values = [values[perm[weights[i]]] for i in range(len(weights))]
    
    # base case: V[i] = 0 for i < weight[0]
    if not weights:
        return 0
    V = [0]*(weights[-1])
    
    # inductive
    for i in range(1, len(weights)):
        for j in range(weight[i-1], weights[i]):
            V[j] = max(V[j-weight[k]] + values[k]
                       for k in range(i), default=0)
    # len(V) = weights[-1]
    for i in range(weights[-1], capacity+1):
        V.append(max(V[-weight[j]] + values[j]
                 for j in range(len(weights))))
        V.popleft()
    return V[-1]

def knapsack_without_repetition(weights, values, capacity):
    # V[i][j] = weight[:i+1], cap j
    # V[i+1][j] = V
    del weights[bisect_right(weights, capacity):]
    del values[len(weights):]
    V = [0]*(capacity+1)
    for i in len(weights):
        W = (V[:weights[i]] +
             [max(V[j-weights[i]] + values[i], V[j])
              for j in range(weights[i], len(V))])
        V = W
    return V[-1]

def shortest_reliable_paths(num_vertices, max_edges, source, edges):
    """For all vertices find the shortest path from source with at most
       max_edges edges.  Return length and sequence of each path."""
    dist = [[float('inf')]*num_vertices]
    dist[0][source] = 0
    prev = [[None]*num_vertices]
    for _ in range(max_edges):
        next_dist = []
        next_prev = []
        for v in range(num_vertices):
            e = min(edges[v],
                    key=lambda e: dist[-1][e.source] + e.length,
                    default=None)
            if e is None:
                next_dist.append(float('inf'))
                next_prev.append(None)
            else:
                next_dist.append(dist[-1][e.source] + e.length)
                next_prev.append(e.source)
        dist.append(next_dist)
        prev.append(next_prev)
    path_num_edges = [min(range(num_vertices), key=lambda i: dist[i][v])
                      for v in range(num_vertices)]
    path_lengths = [dist[path_num_edges[v]][v] for v in range(num_vertices)]
    paths = []
    for v in range(num_vertices):
        path = []
        u = v
        for i in (path_num_edges[v], 0, -1):
            path.append(u)
            u = prev[i][v]
        paths.append(path[::-1])
    return path_lengths, paths


def all_pairs_shortest_paths(n, edges): # Floyd-Warshall O(V**3)
    """Find the shortest path between all vertices."""
    # dist[i][j][k] = length of shortest path between i and j
    #                 with intermediate nodes 0...k-1
    dist = [[float('inf')]*n for _ in range(n)]
    path = [[[] for _ in range(n)] for _ in range(n)]
    for i,j,w in edges:
        dist[i][j] = w
    for k in range(1, n):
        distk = [list(x) for x in dist]
        pathk = [[x for x in y] for y in path]
        for i in range(n):
            for j in range(n):
                d = dist[i][k-1] + dist[k-1][j]
                if d < dist[i][j][k]:
                    distk[i][j] = d
                    pathk[i][j] = path[i][k-1] + [k-1] + path[k-1][j]
        dist = distk
        path = pathk
    return dist, path
    

def combinations(a, k):
    if k > len(a):
        return []
    if not k:
        return [()]
    result = ((i,) for i in range(len(a)-k+1))
    for i in range(k-1, 0, -1):
       result = (x+(j,) for x in result for j in range(x[-1]+1, len(a)-i+1))
    return (tuple(a[i] for i in x) for x in result)


def permutations(a, k):
    if k > len(a):
        return []
    if not k:
        return [()]
    result = (((),-1),)
    for i in range(k, 0, -1):
        result = ((x[0][:s]+(j,)+x[0][s:], j)
                  for x in result
                  for j in range(x[1]+1, len(a)-i+1)
                  for s in range(len(x[0])+1))
    return (tuple(a[i] for i in x[0]) for x in result)


def travelling_salesman(n, edges):
    """Find shortest tour starting at 0."""
    if not n:
        return float('inf'), ()
    length = [[0]*n for _ in range(n)]
    temp = [set() for _ in range(n)]
    for i,j,w in edges:
        length[i][j] = w
        temp[i].add(j)
    edges = temp

    def ts_subset(n):
        if not n:
            return []
        result = [(0,)]
        yield(result)
        for _ in range(2, n+1):
            result = [x+(i,) for x in result for i in range(x[-1]+1, n)]
            yield(result)
            yield(result)

    subsets = ts_subset(n)

    # C[(S,j)] = shortest tour of subset S starting at 0 and ending at j
    C = {(S, 0): (0, (0,)) for S in next(subsets)}

    for s in range(2, n+1):
        B = C
        C = {(S, 0): (float('inf'), ()) for S in next(subsets)}
        for S in next(subsets): # subsets of size s > 1 are yielded twice
            for j in range(1, len(S)):
                Sj = S[:j] + S[j+1:]
                i = min((i for i in Sj if S[j] in edges[i]),
                        key=lambda i: B[(Sj, i)][0] + length[i][S[j]],
                        default=None)
                if i is not None:
                    val, path = B[(Sj, i)]
                    val += length[i][S[j]]
                    path += (S[j],)
                else:
                    val = float('inf')
                    path = ()
                C[S, S[j]] = (val, path)
    j = min((j for j in range(n) if 0 in edges[j]),
            key=lambda j: C[(tuple(range(n)), j)][0] + length[j][0],
            default=None)
    if j is not None:
        val, path = C[(tuple(range(n)), j)]
        val += length[j][0]
    else:
        val = float('inf')
        path = ()
    return val, path


def contiguous_subsequence_max_sum(a):
    curr, result = 0, 0
    for x in a:
        curr = max(curr+x, x)
        result = max(result, curr)
    return result


def road_trip(hotels, daily, f):
    """Destination in miles is hotels[-1]. Ideal daily mileage is daily.
       Daily mileage penalty function is f. Minimize penalty."""
    # P[0] = 0, P[i] = destination hotels[i-1]
    hotels = [0] + hotels
    P = [0]
    prev = [None]
    for i in range(1, len(hotels)):
        j = min(range(i),
                key=lambda j: min(P[j] + f(hotels[i] - hotels[j]))
        P.append(P[j] + f(hotels[i] - hotels[j]))
        prev[i] = j
    i = len(hotels)-1
    path = []
    while i:
        path.append(i-1)
        i = prev[i]
    return path[::-1]


def restaurants_in_line(locations, profits, dist):
    """Which locations should open with expected profit give
       adjacent locations must be dist apart."""
    # 0 ... l0 ... l1 ... l2
    k = bisect_left(locations, dist)
    if k = len(locations):
        return max(profits)
        
    P = [0]*(locations[-1]+1)
    
    for i in range(locations[0], locations[1]):
        P[i] = profits[0]
        
    for i in range(1, k):
        pi = max(profits[i], P[locations[i-1]])
        for j in range(locations[i], locations[i+1]):
            P[i] = pi
            
    for i in range(k, len(locations)):
        Q = list(P)
        for j in range(locations[i], len(P)):
            Q[j] = max(P[j-dist] + locations[i], P[j])
        P = Q
    
    return P[-1]


def string_partion_dictionary(s, dictionary):
    """Return all sequences of words in dictionary that parition s."""
    # dictionary is a boolean function
    R = [[] for _ in range(len(s))]
    R.append([None])
    for i in range(len(s)-1, -1, -1):
        R[i] = [j for j in range(i+1, len(s)+1)
                if s[i:j] in dictionary and R[j]]
    result = []
    if not R[0]:
        return result
    stack = [(0, [])]
    while stack:
        i, path = stack.pop()
        if i == len(s):
            result.append(path)
        else:
            for j in R[i][:-1]:
                stack.append((j, list(path) + [s[i:j]]))
            stack.append((R[-1], path + [s[i:R[-1]]]))
    return result
    

def pebbling_checkboard(board):
    """Given 2d int array, select a set of entries that maximizes their sum
       under constraint entries must not be horizontally or vertically adj."""
    m, n = len(board), len(board[0])
    
    wo_last = [()]
    w_last = []
    for i in range(n):
        wo_last, w_last = wo_last + w_last, [x + (i,) for x in wo_last]
    col_types = w_last + wo_last
    
    col_sets = [set(x) for x in col_types]
    compatible = [[False]*len(col_types) for _ in range(len(col_types))]
    for i in range(len(col_types)):
        for j in range(i+1, len(col_types)):
            if not col_sets[i].union(col_sets[j]):
                compatible[i][j] = True
                compatible[j][i] = True
    
    prev_type, curr_type = 0, 0
    prev_sum, curr_sum = 0, 0
    for t in range(1, len(col_types)): # col_types[0] == ()
        s = sum(board[0][x] for x in col_types[t])
        if s > curr_sum
            curr_type, curr_sum = t, s
    prev_col = [None]
    best_type = [curr_type]
    
    for i in range(1, m):
        next_type = 0
        next_sum = curr_sum
        col = None
        for t in range(1, len(col_types)):
            if compatible[prev_type][t] or compatible[curr_type][t]:
                s = sum(board[i][x] for x in col_types[t])
                if compatible[prev_type][t] and s + prev_sum > curr_sum:
                    next_type, next_sum, col = t, s + prev_sum, -2
                if compatible[curr_type][t] and s + curr_sum > next_sum:
                    next_type, next_sum, col = t, s + curr_sum, -1
        best_type.append(next_type)
        prev_col.append(col)
        prev_sum, curr_sum = curr_sum, next_sum
        prev_type, curr_type = curr_type, next_type
    
    positions = []
    i = m-1
    while i is not None:
        postions.extend((i, j) for j in col_types[best_type[i]])
        i += prev_col[i]
    return curr_sum, positions


from collections import defaultdict

def expression_parenthesize_single_operator(express, op):
    """Return all possible evaluations of expression."""
    # abbca -> ((ab)(bc))a
    
    # P[i][j] = result at express[i:j+1]
    #         = P[i][k] x P[k+1][j] for k in range(i, j+1)
    
    n = len(express)
    P = [[set() for _ in range(n)] for _ in range(n)]
    for i in range(n):
        P[i][i].add(express[i])
        
    for j in range(1, n):
        for i in range(n-j):
            # P[i][i+j] = P[i][k] x P[k+1][i+j] for k in range(i, i+j+1)
            for k in range(i, i+j):
                P[i][i+j].add([op(x, y) for x in P[i][k] for y in P[k+1][i+j]])
                
    return list(P[0][-1])


def longest_palindromic_subsequence(s):
    # P[i][j] = P[i+1][j-1] and s[i] == s[j]
    if not s:
        return 0, ['']
    max_odd_length = 1
    max_odd_seq = [(i, i+1) for i in range(len(s)-1)]
    odd = [True]*len(s)
    for i in range(len(s)//2):
        # odd[j] == True implies s[j-i:j+i+1] is palindrome
        max_odd_seq_new = []
        for j in range(i+1, len(s)-i-1):
            if odd[j] and s[j-i-1] == s[j+i+1]:
                max_odd_seq_new.append((j-i-1, j+i+2))
            else:
                odd[j] = False
        if not max_odd_seq_new:
            break
        max_odd_seq = max_odd_seq_new
        max_odd_length += 2
    
    even = []
    max_even_seq = []
    max_even_length = 0
    for i in range(len(s)-1):
        if s[i] == s[i+1]:
            even.append(True)
            max_even_seq.append((i, i+2))
        else:
            even.append(False)
    if max_even_seq:
        max_even_length = 2
    for i in range((len(s)-1)//2):
        # even[j] == True implies s[j-i:j+i+2] is palindrome
        max_even_seq_new = []
        for j in range(i+1, len(s)-i-2):
            if even[j] and s[j-1] == s[j+i+2]:
                max_even_seq_new.append((j-1, j+i+3))
            else:
                even[j] = False
        if not max_even_seq_new:
            break
        max_even_seq = max_even_seq_new
        max_even_length += 2
    
    if max_odd_length > max_even_length:
        return max_odd_length, [s[x[0]:x[1]] for x in max_odd_seq]
    else:
        return max_even_length, [s[x[0]:x[1]] for x in max_even_seq]


def longest_palindromic_subsequence(s):
    # P[i][j] = P[i+1][j-1] and s[i] == s[j]
    if not s:
        return 0, ['']

    odd = [0]*len(s)
    # odd[i] == j implies s[i-j:i+j+1] is palindrome of length 2*j+1
    for j in range(1, len(s)//2+1):
        change = False
        for i in range(j, len(s)-j):
            if odd[i] == j-1 and s[i-j] == s[i+j]:
                odd[i] = j
                change = True
        if not change:
            break

    even = [1 if s[i] == s[i+1] else 0 for i in range(len(s)-1)]
    # even[i] = j implies s[i-j+1:i+j+1] is palindrome of length 2*j
    for j in range(2, len(s)//2+1):
        change = False
        for i in range(j-1, len(s)-j-1):
            if even[i] == j-1 and s[i-j+1] == s[i+j]:
                even[i] = j
                change = True
        if not change:
            break

    m = max(odd)
    n = max(even, default=0)
    if m >= n:
        return 2*m+1, [s[i-m:i+m+1] for i in range(len(s)) if odd[i] == m]
    return 2*n, [s[i-n+1:i+n+1] for i in range(len(s)-1) if even[i] == n]


def longest_common_substring(s, t):
    if not s or not t:
        return 0, []
    max_length = 0
    max_string = set()

    # L[i][j] = 0 if s[i] != t[j]
    #           1 + L[i-1][j-1] if s[i] == t[j]
    
    L = [int(s[0] == t[j]) for j in range(len(t))]
    if any(L):
        max_length = 1
        max_string.add(s[0])

    N = [int(s[i] == t[0]) for i in range(len(s))]
    if any(N):
        max_length = 1
        max_string.add(t[0])

    for i in range(1, len(t)):
        M = N[i:i+1] + L[1:]
        for j in range(1, len(t)):
            if s[i] == t[j]:
                M[j] = 1 + L[j-1]
                if M[j] > max_length:
                    max_length = M[j]
                    max_string = set([s[i-max_length+1:i+1]])
                elif M[j] == max_length:
                    max_string.add(s[i-max_length+1:i+1])
        L = M

    return max_length, list(max_string)
    

def primitive_string_splits(cuts, length):
    """With a string of given length, order cuts that minimizes costs, where each split cost the length of the string."""
    # string: 0 1 2 3 4 5     cuts: 1, 3
    #        0 1 2 3    4 5   cost: = 6
    #       0 1   2 3   4 5        += 4
    
    #     0   1   2   3   4
    #  ---------------------
    # 0| 0-0 0-1 0-2 0-3 0-4
    # 1|     1-1 1-2 1-3 1-4
    # 2|         2-2 2-3 2-4
    # 3|             3-3 3-4
    # 4|                 4-4

    if not cuts:
        return 0
    if cuts[0]:
        cuts = [0] + cuts
    if cuts[-1] != length:
        cuts.append(length)

    C = [[0]*len(cuts) for j in range(len(cuts))]
    order = [[[] for _ in range(len(cuts))] for j in range(len(cuts))]
    # C[i][j] = cost of spitting string[cuts[i]:cuts[j]] at cuts[i+1:j]
        
    for i in range(len(cuts)-2):
        C[i][i+2] = cuts[i+2] - cuts[i]
        order[i][i+2].append(i+1)
    
    for j in range(3, len(cuts)):
        for i in range(len(cuts)-j):
            # string[cuts[i]:cuts[i+j]] at cuts[i+1:i+j]
            # i --------- i+j
            # i -- k k -- i+j
            k = min(range(i+1, i+j), key=lambda k: C[i][k] + C[k][i+j])
            C[i][i+j] = cuts[i+j] - cuts[i] + C[i][k] + C[k][i+j]
            order[i][i+j] = cuts[k:k+1] + order[i][k] + order[k][i+j]
    return C[0][-1], order[0][-1]


def counting_heads(prob, n, k):
    """Given probability of heads for n biased coins, find the probability
       of flipping all coins and obtaining exactly k heads."""
    # P[i][j] = counting_heads(prob[:i], i, j)
    #         = 1 if i = j = 0
    #         = 0 if j > i
    #         = prob[i]*P[i-1][j-1] + (1-prob[i])*P[i-1][j]

    if  k > n:
        return 0
    qrob = [1-x for x in prob]
    P = [1] + [0]*k
    
    for i in range(1, n-k+1):
        P = P[0:1] + [prob[i-1]*P[j-1] + qrob[i-1]*P[j]
                      for j in range(1, k+1)]
        P[0] *= qrob[i-1]
        
    for i in range(n-k+1, n+1):
        P = [0]*(i-n+k) + [prob[i-1]*P[j-1] + qrob[i-1]*P[j]
                           for j in range(i-n+k, k+1)]
                           
    return P[-1]


def longest_common_subsequence_length(s, t):
    if not s or not t:
        return 0, ['']
    L = [0]*len(t)
    for j in range(len(t)):
        if s[0] == t[j]:
            for k in range(j, len(t)):
                L[k] = 1
            break
    for i in range(1, len(s)):
        M = [0]*len(t)
        M[0] = max(L[0], int(s[i] == t[0]))
        for j in range(1, len(t)):
            M[j] = max(M[j-1], L[j-1] + int(s[i] == t[j]), L[j])
        L = M
    return L[-1]


def longest_common_subsequence(s, t):
    if not s or not t:
            return 0, []
    L = [[0]*len(t) for _ in range(len(s))]
    prev = [[[] for _ in range(len(t))] for _ in range(len(s))]
    for j in range(len(t)):
        if s[0] == t[j]:
            L[0][j] = 1
            prev[0][j] += [(1,1)]
            for k in range(j+1, len(t)):
                L[0][k] = 1
                prev[0][k] += [(0, 1)]
            break
        prev[0][j] += [(0, 1)]

    for i in range(1, len(s)):
        L[i][0] = max(L[i-1][0], int(s[i] == t[0]))
        prev[i][0] += [(1, 0)]
        for j in range(1, len(t)):
            if s[i] == t[j]:
                L[i][j] = L[i-1][j-1] + 1
                prev[i][j] += [(1, 1)]
            elif L[i][j-1] > L[i-1][j]:
                L[i][j] = L[i][j-1]
                prev[i][j] += [(0, 1)]
            elif L[i][j-1] < L[i-1][j]:
                L[i][j] = L[i-1][j]
                prev[i][j] += [(1, 0)]
            else:
                L[i][j] = L[i][j-1]
                prev[i][j] += [(1, 0), (0, 1)]

    max_indices, max_length = [], 0
    for x in range(len(t)):
        if L[-1][j] > max_length:
            max_indices = [(len(s)-1, j)]
            max_length = L[-1][j]
        elif L[-1][j] == max_length:
            max_indices += [(len(s)-1, j)]

    sequences = set()
    stack = [(i, j, [])]
    while stack:
        i, j, q = stack.pop()
        if i < 0:
            sequences.add(''.join(q[::-1]))
            continue
        for di,dj in prev[i][j]:
            if di == dj == 1:
                r = q + [s[i]]
            else:
                r = q
            stack += [(i - di, j - dj, r)]
    return max_length, list(sequences)


def card_stack_sum(cards):
    """Two players alternate taking the top or bottom card in a deck, add its
       value to their total. Find the optimal strategy."""
    n = len(cards)
    
    sums = [0] + cards
    for i in range(1, len(cards)+1):
        sums[i] += sums[i-1]
    # sum(cards[i:j+1]) = sums[j+1] - sums[i]
    
    values = [[None for _ in range(n)] for _ in range(n)]
    choices = [[None for _ in range(n)] for _ in range(n)]
    # values[i][j] = minimal total guaranteed from cards[i:j+1]
    # choices[i][j] = optimal card choice from cards[i:j+1]
    
    # base case
    # deck of size 1
    for i in range(n):
        values[i] = cards[i]
        choices[i] = i
    
    # induction
    # values[i][j] = sum(cards[i:j+1]) - min(values[i+1][j], values[i][j-1])
    # choices[i][j] = i if values[i+1][j] < values[i][j-1] else j
    for j in range(1, n):
        for i in range(n-j):
            # compute values[i][i+j]
            if values[i+1][i+j] < values[i][i+j-1]:
                values[i][i+j] = sums[j+1] - sums[i] - values[i+1][i+j]
                choices[i] = i
            else:
                values[i][i+j] = sums[j+1] - sums[i] - values[i][i+j-1]
                choices[i][i+j] = i+j
    
    return values, choices
    

def cutting_cloth(length, width, cloth):
    """Start with cloth of dimension length by width. Split horizontally or
       vetically to produce smaller cloths of given dimensions and value."""
    # cloth[i] = (length, width, value)
    
    # V[i][j] = max value from cloth of dimension i by j
    V = [[0]*(width+1) for _ in range(length+1)]
    C = [[(0,0)]*(width+1) for _ in range(length+1)]
    cloth = [c for c in cloth if c[0] <= length and c[1] <= width]
    
    for x,y,w in cloth:
        V[x][y] = max(V[x][y], w)
        V[y][x] = max(V[y][x], w)
        
    m = min(min(c[0] for c in cloth), min(c[1] for c in cloth))
    for i in range(m, length+1):
        for j in range(m, width+1):
            k = (m+i)//2
            cuts = [V[x][j]+V[i-x][j] for x in range(m, k+1)]
            x = max(range(m, k+1), key=lambda x: cuts[x])
            if cuts[x] > V[i][j]:
                V[i][j] = cuts[x]
                C[i][j] = (x,0)
            k = (m+j)//2
            cuts = [V[i][y]+V[i][j-y] for y in range(m, k+1)]
            h = max(range(m, k+1), key=lambda y: cuts[y])
            if cuts[y] > V[i][j]:
                V[i][j] = cuts[y]
                C[i][j] = (0,y)
                
    max_value = V[length][width]
    cuts = []
    stack = [(length, width)]
    while stack:
        i,j = stack.pop()
        x,y = C[i][j]
        if not x and not y:
            continue
        cuts.append((i,j,x,y))
        if x:
            stack.extend([(x,j),(i-x,j)])
        else:
            stack.extend([(i,y),(i,j-y)])
    return max_value, cuts


def two_teams_race_to_n(p, n):
    """Probability p of winning any given game, 1-p for opponent.  Probability of winning n games given win loss record."""
    P = [[0 for _ in range(n+1)] for _ in range(n+1)]
    for j in range(n):
        P[n][j] = 1
    for i in range(n):
        P[i][n] = 0
    for i in range(n-1, -1, -1):
        for j in range(n-1, -1, -1):
            P[i][j] = p*P[i+1][j] + (1-p)*P[i][j+1]
    return P


def garage_sale(n, edges, values):
    """Hamiltonian cycle that maximizes value at each node minus transportation cost."""
    if not n:
        return float('inf'), ()
    length = [[0]*n for _ in range(n)]
    temp = [set() for _ in range(n)]
    for i,j,w in edges:
        length[i][j] = w
        temp[i].add(j)
    edges = temp

    def ts_subset(n):
        if not n:
            return []
        result = [(0,)]
        yield(result)
        for _ in range(2, n+1):
            result = [x+(i,) for x in result for i in range(x[-1]+1, n)]
            yield(result)
            yield(result)

    subsets = ts_subset(n)

    # C[(S,j)] = value, path, value-length[j][0] of shortest tour of subset S starting at 0 and ending at j,
    C = {(S, 0): (0, (0,), 0) for S in next(subsets)}
    max_value = 0
    max_path = ()

    for s in range(2, n+1):
        B = C
        C = {(S, 0): (0, (), 0) for S in next(subsets)}
        for S in next(subsets): # subsets of size s > 1 are yielded twice
            for j in range(1, len(S)):
                Sj = S[:j] + S[j+1:]
                i = max((i for i in Sj if S[j] in edges[i]),
                        key=lambda i: B[(Sj, i)][0]+values[S[j]]-length[i][S[j]],
                        default=None)
                if i is not None:
                    val, path = B[(Sj, i)]
                    val += values[S[j]] - length[i][S[j]]
                    path += (S[j],)
                    cycle = val - length[S[j]][0]
                else:
                    val = float('inf')
                    path = ()
                    cycle = 0
                C[S, S[j]] = (val, path, cycle)
        maxS = max(C, key=lambda k: C[k][2])
        if C[maxS] > max_value:
            max_value = C[maxS][2]
            max_path = C[maxS][1]

    j = min((j for j in range(n) if 0 in edges[j]),
            key=lambda j: C[(tuple(range(n)), j)][2],
            default=None)
    if j is not None:
        _, path, cycle = C[(tuple(range(n)), j)]
    else:
        path = ()
        cycle = 0
    if cycle > max_value:
       max_value = cycle
       max_path = path
    return max_value, max_path


def make_change_possible(coins, value):
    """Return if multiset of coins (infinite multiplicity) sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]

    if not coins:
        return False
    coins.append(value+1)

    poss = [False]*(value+1)
    poss[0] = True

    for i in range(len(coins)-1):
        for j in range(coins[i], coins[i+1]):
            poss[j] = any(poss[j-c] for c in coins[:i+1])

    return poss[-1]


def make_change(coins, value):
    """Return list of coins (repeats allowed) that sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]
    if not coins:
        return []
    coins.append(value+1)

    last = [[] for _ in range(value+1)]
    last[0] = [None]

    for i in range(len(coins)-1):
        for j in range(coins[i], coins[i+1]):
            last[j] = [c for c in coins[:i+1] if last[j-c]]
            
    sequences = set()
    stack = [(value, [])]
    while stack:
        v, s = stack.pop()
        if not v:
            sequences.add(tuple(sorted(s)))
            continue
        for c in last[v]:
            stack.append((v-c, s+[c]))

    return sorted(sequences)


def make_change_without_repetition_possible(coins, value):
    """Return if there is subset of coins that sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]
    if not coins:
        return False

    possible = [False]*coins[0]
    possible[0] = True

    for c in coins:
        temp = possible[:c]
        temp.extend([possible[x] or possible[x-c]
                     for x in range(c, len(possible))])
        stop = min(len(possible)+c, value+1)
        temp.extend([possible[x-c]
                     for x in range(len(possible), stop)])
        possible = temp

    return possible[value]


# ****** finished prev 2d
def make_change_without_repetition(coins, value):
    """Return permutation of subset of coins that sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]
    if not coins:
        return False

    prev = [[[] for j in range(coins[0])]]
    for i in range(len(coins)):
        prev[i][0] = [None]

    for i in range(len(coins)):
        prev[i] = prev[i-1][:coins[i]]
        prev[i].extend([prev[i-1][x] + [c] if prev[i-1][x-c] else prev[i-1][x]
                       for x in range(c, len(prev[i-1]))])
        stop = min(len(prev)+c, value+1)
        temp.extend([prev[i-1][x] + [c] if prev[i-1][x-c] else []
                     for x in range(len(prev), stop)])
        prev = temp

    result = set()
    stack = [(value, [])]
    while stack:
        v, c = stack.pop()
        if not v:
            result.add(tuple(sorted(c)))
            continue
        stack.extend((v-x, c + [x]) for x in prev[v])
    return sorted(result)


def make_change_at_most_possible(coins, value, max_coins):
    """Return if multiset of coins (infinite multiplicity) of size max_coins
       that sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]
    if not coins:
        return False
    coins = [0] + coins + [value+1]
        
    # N[i] == number of coins to make change for i
    N = [float('inf')]*(value+1)
    N[0] = 0
    
    for i in range(1, len(coins)-1):
        for j in range(coins[i], coins[i+1]):
            N[j] = min(N[j-c] for c in coins[:i+1]),
    
    return N[value] <= max_coins


def make_change_at_most(coins, value, max_coins):
    """Return if multiset of coins (infinite multiplicity) of size max_coins
       that sum to value."""
    coins.sort()
    del coins[bisect_right(coins, value):]
    if not coins:
        return False
        
    # N[i] == number of coins to make change for i
    N = [[[]] for _ in range(value+1)]
    for c in coins:
        N[c] = [[c]]
    
    for c in coins:
        for i in range(c, value+1):
            if not N[i-c][0]:
                continue
            elif not N[i][0] or len(N[i-c][0])+1 < len(N[i][0]):
                N[i] = [N[i-c] + [c]]
            elif len(N[i-c][0])+1 == len(N[i][0]):
                N[i].append(N[i-c])
    
    if not N[value] or len(N[value]) > max_coins:
        return False, []
    return True, N[value]
    

def optimal_binary_search_tree(words, freq):
    """Construct binary search tree to minimize cost of search, each node
       access costs one unit, words are found with given frequency"""
    # words are in sorted order

    class Node:

        def __init__(self, word):
            self.word = word
            self.left = None
            self.right = None

    # cumsum[i] = sum(freq[:i])
    # sum(freq[i:j]) = cumsum[j] - cumsum[i]
    cumsum = [0]
    for f in freq:
        cumsum.append(cumsum[-1] + f)

    n = len(words)

    # root[i][j] = index of root word for tree of words[i:j]
    root = [[None for _ in range(n+1)] for _ in range(n+1)]
    cost = [[0]*(n+1) for _ in range(n+1)]

    # cost[i][j] = sum[i][j] + min(c[i][k-1]+c[k+1][j] for k in range(i, j+1))
    for j in range(1, n+1):
        for i in range(n-j+1):
            # compute cost, tree for [i][i+j]
            k = min(range(i, i+j),
                    key=lambda k: cost[i][k] + cost[k+1][i+j],
                    default=i)
            root[i][i+j] = k
            cost[i][i+j] = cumsum[i+j]-cumsum[i] + cost[i][k]+cost[k+1][i+j]

    # construct tree
    def treerange(i, j):
        k = root[i][j]
        if k is None:
            return None
        node = Node(words[k])
        node.left = treerange(i, k)
        node.right = treerange(k+1, j)
        return node

    return treerange(0, n)
    

def smallest_vertex_cover():
    pass


def subset_sum(nums, total):
    """Return if some subset of nums sums to total."""
    # same as make_change_possible
    if not total:
        return True
    nums.sort()
    del nums[bisect_right(nums, total):]
    if not nums:
        return False
    
    S = [False]*(nums[0])
    S[0] = True
    
    for x in nums:
        temp = S[:x]
        temp.extend(S[i] or S[i-x] for i in range(x, len(S)))
        stop = min(total+1, len(S)+x)
        temp.extend(S[i-x] for i in range(len(S), stop))
        S = temp
    
    return nums[total]
    
    S = set([0])
    prev = 0
    curr = 1
    while curr != prev and total not in S:
        S.update([x+y for x in S for y in nums if x+y <= total])
        curr, prev = len(S), curr
    return total in S
    
    
def machine_reliability_redundancy(prob, cost, budget):
    """Each machine completes task with (independent) probability of success and cost.  Maximize probability of completing all task given budget."""
    n = len(prob)
    s = sum(cost)
    if s > budget:
        return [0]*n
    budget -= s
    
    temp = sorted(range(i), key=lambda i: cost[i])
    prob = [prob[i] for i in temp]
    cost = [cost[i] for i in temp]
    del cost[bisect_right(cost, budget):]
    if not cost:
        return [1]*n
    del prob[len(cost):]
    remainder = [1]*(n-len(cost))
    n = len(cost)
    
    # B[i][j] == num machines i at total j
    # return B[budget] + remainder
    B = [[1]*n for _ in range(cost[0])]
    
    # R[i][j] == (1-prob[i])*B[i][j] == probability of machine i failing
    R = [[1-x for x in prob] for _ in range(cost[0])]
    
    def marg_rel(f, p):
        # marginal increase in reliability i.e. log of probability of sucess
        # f = (1-p)**m i.e. prob of all m copies of machine failing
        return log((1-(1-p)*f)/(1-f))
    
    cost.append(budget)
    for k in range(len(cost)-1):
        for i in range(cost[k], cost[k+1]):
            j = max(range(k+1),
                    key=lambda j: marg_rel(R[i-cost[j]][j], cost[j])
            B.append(list(B[i-cost[j]]))
            B[-1][j] += 1
            R.append(R[i-cost[j]])
            R[-1][j] *= (1-prob[j])
    
    return B[budget] + remainder


def three_partition_equal_sums(a):
    """Return if sequence can be partitioned into three parts of equal sum."""
    # f((x,y,z), a[:i]) = any(f((x-a[i-1], y, z), a[:i-1])
    #                         f((x, y-a[i-1], z), a[:i-1])
    #                         f((x, y, z-a[i-1]), a[:i-1]))
    # f((0,0,0), a[:0]) = True
    # f((-x,y,z), a[:i]) = False
    
    s = sum(a)
    s, temp = divmod(s, 3)
    if temp:
        return False
        
    P = [[[False]*(s+1) for _ in range(s+1)] for _ in range(s+1)]
    P[0][0][0] = True
    
    for num in a:
        temp = list(list(list(y) for y in x) for x in P)
        for x in range(num, s+1):
            for y in range(num):
                for z in range(num):
                    temp[x][y][z] = temp[x][y][z] or a[x-num][y][z]
        for x in range(num):
            for y in range(num, s+1):
                for z in range(num):
                    temp[x][y][z] = temp[x][y][z] or a[x][y-num][z]
        for x in range(num):
            for y in range(num):
                for z in range(num, s+1):
                    temp[x][y][z] = temp[x][y][z] or a[x][y][z-num]
        for x in range(num, s+1):
            for y in range(num, s+1):
                for z in range(num, s+1):
                    temp[x][y][z] = (temp[x][y][z] or a[x-num][y][z] or
                                     a[x][y-num][z] or a[x][y][z-num])
        P = temp
    
    return P[s][s][s]


def sequence_alignment_gap_score(s, t, delta, gap_penalty):
    """Add gap insertions to align sequnces and maximize delta score."""
    # score('ab-c', 'acd-') = sum(delta[s'[i], t'[i]] for i in range(4))
    
    # S[i][j] = max score of s[:i] and t[:j]
    S = [0]
    for x in t:
        S.append(delta['-', x])
    
    def editor(i, j, x):
        a = '-' if not x[0] else s[i-1]
        b = '-' if not x[1] else t[j-1]
        return (a, b)
        
    edit_type = [(0,-1), (1-,0), (-1,-1)]
    
    def score(i, j, S, Si, E, Ei):
        x = delta['-', t[j-1]] + Si[j-1] + int(Ei[j-1]==(0,-1))*gap_penalty
        y = delta[s[i-1], '-'] + S[j] + int(E[j]==(-1,0))*gap_penalty
        z = delta[s[i-1], t[j-1]]
        temp = [x,y,z]
        e = max(range(3), key=lambda e: score[e])
        return temp[e], e
        
    # E[i][j] = edit_type index of last entry in s[:i] and t[:j] alignment
    E = [(-1,-1)] + [(0,-1)]*len(t)
    
    for i in range(1, len(s)+1):
        Si = [delta[s[i-1], '-'] + S[0]]
        Ei = [(1,0)]
        for j in range(1, len(t)+1):
            s, e = score(i, j, S, Si, E, Ei)
            Si.append(s)
            Ei.append(e)
        S = Si
        E.append(Ei)
    
    score = S[-1]
    result = []
    i = len(s)
    j = len(t)
    while i > -1:
        result.append(editor(E[i][j]))
        i -= E[i][j]
        j -= E[i][j]
    
    se = ''.join(x[0] for x in result)
    te = ''.join(x[1] for x in result)
    return S[-1], se, te


def local_sequence_alignment(s, t, delta):
    """Return substring of s and t with max sequence alignment score."""
    # score('ab-c', 'acd-') = sum(delta(s'[i], t'[i]) for i in range(4))

    # S[i][j] = max score of suffixes of s[:i] and t[:j]
    # P[i][j] = prev index
    S = [0]
    P = [[None]*(len(t)+1) for _ in range(len(t)+1)]
    P[0][0] = 0

    def prev_score(i, j, Si=None):
        nonlocal S
        if not i:
            score = [0, S[-1]+delta('-', t[j-1])]
        elif not j:
            score = [0, -float('inf'), -float('inf'), S[0]+delta(s[i-1], '-')]
        else:
            score = [0, Si[j-1]+delta('-', t[j-1]), S[j-1]+delta(s[i-1], t[j-1]),
                     S[j]+delta(s[i-1], '-')]
        k = max(range(len(score)), key=lambda k: score[k])
        return score[k], k

    for j in range(1, len(t)+1):
        ps = prev_score(0, j)
        P[0][j] = max(range(len(ps)), key=lambda k: ps[k])
        S.append(ps[P[0][j]])

    max_stop = (0, max(range(len(t)+1), key=lambda j: S[j]))
    max_score = S[max_stop[1]]

    for i in range(1, len(s)+1):
        score, k = prev_score(i, 0)
        P[i][0] = k
        Si = [score]
        for j in range(1, len(t)+1):
            score, k = prev_score(i, j, Si)
            P[i][j] = k
            Si.append(score)
        S = Si
        j = max(range(len(t)+1), key=lambda j: S[j])
        if max_score < S[j]:
            max_score = S[j]
            max_stop = (i,j)

    edit = [None, lambda i,j: ('-', t[j-1]), lambda i,j: (s[i-1], t[j-1]),
            lambda i,j: (s[i-1], '-')]
    prev_pos = [None, lambda i,j: (i,j-1), lambda i,j: (i-1,i-1),
                 lambda i,j: (i-1,j)]
    seq = []
    i,j = max_stop
    while P[i][j]:
        k = P[i][j]
        seq.append(edit[k](i,j))
        i,j = prev_pos[k](i,j)
    se, te = [''.join(x)[::-1] for x in zip(*seq)]
    return max_score, se, te
    

def weighted_intervals_bisect(intervals):
    """Given left, right endpoints and weights of each interval
       return mutually disjoint subset that maximizes total weight."""
    # interval[i] == (li, ri, wi) including li and ri

    intervals.sort(key=lambda x: x[1])
    rights = [x[1] for x in intervals]

    # W[i] = max weight of subset of intervals[:i]
    # I[i] = subset of optimal weight of intervals[:i]
    W = [0]
    I = [[]]

    for i in range(1, len(intervals)+1):
        j = bisect_left(rights[:i-1], intervals[i-1][0])
        w = W[j] + intervals[i-1][2]
        if w > W[i-1]:
            W.append(w)
            I.append(I[j] + [i-1])
        else:
            W.append(W[-1])
            I.append(I[-1])

    return W[-1], [intervals[i] for i in I[-1]]
    

# edit distance divide conquer