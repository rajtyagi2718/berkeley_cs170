class UndirectedGraph:
    
    def __init__(self, numvertices):
        self._adjList = [[] for _ in range(numvertices)]
        self.numEdges = 0
    
    @property
    def numvertices(self):
        return len(self._adjList)
    
    def addEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if i > j:
            i, j = j, i
        self._addEdge(i, j)
    
    def _addEdge(self, i, j):
        self._adjList[i].append(j)
        self.numEdges += 1
    
    def hasEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if i > j:
            i, j = j, i
        self._hasEdge(i, j)
    
    def _hasEdge(self, i, j):
        return j in self._adjList[i]
    
    def removeEdge(self, i, j):
        self._inBounds(i); self._inBounds(j)
        if not self.hasEdge(i, j):
            raise ValueError('Graph has no edge between {} and {}'.format(i, j))
        if i > j:
            i, j = j, i
        self._removeEdge(i, j)
    
    def _removeEdge(self, i, j):
        adjI = self._adjList[i]
        k = adjI.index(j)
        adjI[k] = adjI[-1]
        adjI[].pop()
        self.numEdges -= 1
    
    def degree(self, i):
        smaller = sum(self._hasEdge(j, i) for j in range(i))
        larger = len(self._adjList[i])
        return smaller + larger
    
    def kthEdge(self, i, k):
        if k < 0:
            raise IndexError('kthEdge starts with index 0 i.e. first edge')
        iDeg = self.degree(i)
        if k > iDeg:
            raise IndexError('Vertex {} has less than {} edges'.format(i, k))
        largerDeg = len(self._adjList[i])
        smallerDeg = iDeg - largerDeg
        if k > smallerDeg:
            return self._adjList[i][k - smallerDeg - 1]
        return self._smallerKthEdge(i, k)
    
    def _smallerKthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 1:
                    return (i, j)
                k -= 1
    
    def _kthEdge(self, i, k):
        for j in range(i):
            if self._hasEdge(j, i):
                if k == 1:
                    return (i, j)
                k -= 1
        return self._adjList[i][k-1]
    
    def listEdges(self, i):
        result = [j for j in range(i) if self._hasEdge(j, i)]
        result.extend(self._adjList[i])
        return result
    
    def neighbors(self, i):
        return self.listEdges(i)
    
    def addVertex(self):
        self._adjList.append([])

    def _inBounds(self, i):
        if i < 0:
            raise IndexError('Graph edge index {} cannot be negative'.format(i))
        if i >= self.numvertices:
            raise IndexError('Graph edge index {} must be less than
                              numvertices {}'.format(i, self.numvertices))


class WeightedDirectedGraph:
    
    
    def __init__(self, numvertices):
        self._entries = [[] for _ in range(numvertices)]
        self._exits = [[] for _ in range(numvertices)]
        self.numEdges = 0
    
    @property
    def numvertices(self):
        return len(self._entries)
        
    
    class Edge:
        
        def __init__(self, source, target, weight):
            self.source = source
            self.target = target
            self.weight = weight
    
    
    def addEdge(self, source, target, weight):
        self._inBounds(source); self._inBounds(target)
        self._addEdge(source, target, weight)
    
    def _addEdge(self, source, target, weight):
        E = Edge(source, target, weight)
        self._exits[source].append(E)
        self._entries[target].append(E)
        self.numEdges += 1
        
    
    def hasEdge(self, source, target):
        self._inBounds(source); self._inBounds(target)
        return self._getEdge(source, target) is not None
    
    def _getEdge(self, source, target):
        if len(self._exits[source]) > len(self._entries[target]):
            return self._entries[target][self._getEntryIndex(source, target)]
        return self._exits[target][self._getExitIndex(source, target)]
        
    def _getExitIndex(self, source, target):
        exits = self._exits[source]
        for i in range(len(exits)):
            if exits[i].target == target:
                return i
    
    def _getEntryIndex(self, source, target):
        entries = self.entries[target]
        for i in range(len(entries)):
            if entries[i].source == source:
                return i
                
    
    def changeWeight(self, source, target, weight):
        self._inBounds(source); self._inBounds(target)
        E = self._getEdge(source, target)
        E.weight = weight
    
    
    def listOutEdges(self, source):
        return list(self._exits[source]))
    
    def listInEdges(self, target):
        return list(self._entries[target])
    
    def neighbors(self, source):
        return [E.target for E in self._exits[source]]
    
    def removeEdge(self, source, target):
        if not self.hasEdge(source, target):
            raise ValueError('Graph has no edge
                              from {} to {}'.format(source, target))
        self._inBounds(source); self._inBounds(target)
        self._removeOutEdge(source, target)
        self._removeInEdge(source, target)
    
    def _removeOutEdge(self, source, target):
        exits = self._exits[source]
        i = self._getExitIndex(source, target)
        exits[i] = exits[-1]
        exits.pop()
    
    def _removeInEdge(self, source, target):
        entries = self.entries[target]
        i = self._getEntryIndex(source, target)
        entries[i] = entries[-1]
        entries.pop()
    
    def removeOutEdges(self, source):
        self._inBounds(source)
        self._removeOutEdges(source)
    
    def _removeOutEdges(self, source):
        for E in self._exits[source]:
            self._removeInEdge(source, E.target)
        self._exits[source] = []
    
    def removeInEdges(self, target):
        self._inBounds(target)
        self._removeInEdges(target)
    
    def _removeInEdges(self, target):
        for E in self._entries[target]:
            self._removeOutEdge(E.source, target)
        self._entries[target] = []
    
    def inDegree(self, target):
        return len(self._entries[target])
    
    def outDegree(self, source):
        return len(self._exits[source])
    
    def degree(self, vertex):
        return self.inDegree(vertex) + self.outDegree(vertex)
    
    def _inBounds(self, i):
        if i < 0:
            raise IndexError('Graph vertex {} must be nonnegative'.format(i))
        if i >= self.numvertices:
            raise IndexError('Graph vertex {} must be less than
                              numvertices {}'.format(i, self.numvertices))


class DFSUndirected: # O(V+E)
    
    def __init__(self, graph):
        self.graph = graph
        self._visited = [False] * graph.numvertices
        self._preclock = [None] * graph.numvertices
        self._postclock = [None] * graph.numvertices
        self._ccnum = [None] * graph.numvertices
        self._edgeto = [None] * graph.numvertices
        self._clock = 0
        self._cc = 0
        self._nontreeedges = []
        for v in self.graph.numvertices:
            if not self._visited[v]:
                self._explore(v)
                self._cc += 1
    
    def _explore(self, v):
        self._visited[v] = True
        self._previsit(v)
        for w in self.graph.neighbors(v):
            if not self._visited[w]:
                self._edgeto[w] = v
                self._explore(w)
            elif w != self._edgeto[v]: # self.preclock[w] != self.preclock[v]-1
                # w is ancestor of v (not parent)
                self._nontreeedges.append((w,v))
        self._postvisit(w)
    
    def _previsit(self, v):
        self._ccnum[v] = self._cc
        self._cc += 1
        self._preclock[v] = self._clock
        self._clock += 1
    
    def _postvisit(self, v):
        self._postclock[v] = self._clock
        self._clock += 1
    
    def getPreOrder(self):
        return sorted(range(self.graph.numvertices),
                      key=lambda v: self._preclock[v]))
    
    def getPostOrder(self):
        return sorted(range(self.graph.numvertices),
                      key=lambda v: self._postclock[v]))
    
    def getConnectedComponents(self):
        result = [[] for _ in range(self._cc)]
        for v in range(self.graph.numvertices):
            result[self._ccnum[v]].append(v)
        return result
    
    def getNumCC(self):
        return self._cc
    
    def hasPath(self, v, w):
        return self.getCCNum(v) == self.getCCNum(w)
    
    def getPath(self, v, w):
        if not self.hasPath(v, w):
            return None
        return self._getPath(v, w)
    
    def _getPath(self, v, w):
        vabranch = []
        # get path from v to latest common ancestor
        while self._preclock[v] > self._preclock[w]:
            vabranch.append(v)
            v = self._edgeto[v]
        while self._postclock[v] < self._postclock[w]:
            vabranch.append(v)
            v = self._edgeto[v]
        # get path from latest common ancestor to w
        wabranch = self._getLineage(v, w)
        return vabranch + wabranch
    
    def _getLineage(self, a, v):
        result = deque()
        while v != a:
            result.appendleft(v)
            v = self._edgeto[v]
        result.appendleft(a)
        return list(result)
    
    def hasCycle(self):
        return bool(self._nontreeedges)
    
    def getCycles(self):
        result = []
        for v, w in self._nontreeedges: # self._preclock[v] < self._preclock[w]
            # v is ancestor of w (not parent)
            result.append(self._getLineage(v, w))
        ### how about cycles with more than one nontreeedge?
        return result
        
class DFSDirected: # O(V+E)
    
    def __init__(self, graph):
        self.graph = graph
        self._edgeto = [None] * graph.numvertices
        self._preclock = [0] * graph.numvertices
        self._postclock = [0] * graph.numvertices
        # self._treeedges = [] prev (prew postw postv)
        self._backedges = [] # prew prev (postv postw)
        self._forwardedges = [] # prev prew postw (postv)
        self._crossedges = [] # prew postw prev (postv)
        self._clock = 1
        for v in range(graph.numvertices):
            if not self._preclock[v]:
                self._explore(v)
    
    def _explore(self, v):
        self._previsit(v)
        self._visit(v)
        self._postvisit(v)
    
    def _previsit(self, v):
        self._preclock[v] = self._clock
        self._clock += 1
    
    def _visit(self, v):
        for w in self.graph.neighbors(v):
            if not self._preclock(w):
                self._edgeto[w] = v
                self._explore(w)
            elif not self._postclock[w]:
                self._backedges.append((v,w))
            elif self._preclock[v] < self._preclock[w]:
                self._forwardedges.append((v,w))
            else:
                self._crossedges.append((v,w))
    
    def _postvisit(self, v):
        self._postclock[v] = self._clock
        self._clock += 1
        
    def getPreOrder(self):
        return sorted(range(self.graph.numvertices),
                      key=lambda v: self._preclock[v]))
    
    def getPostOrder(self):
        return sorted(range(self.graph.numvertices),
                      key=lambda v: self._postclock[v]))
    
    def hasCycle(self):
        return bool(self._backedges)
    
    def getTopologicalSort(self): # reverse postOrder
        if self.hasCycle():
            return []
        return sorted(range(self.graph.numvertices),
                      key=lambda v: -self._postclock[v])
        
    def hasPath(self, v, w):
        return self._hasTreePath(v, w) or self._isNonTreeEdge(v, w)
    
    def _hasTreePath(self, v, w):
        return ((self._preclock[v] < self._preclock[w]) and
                (self._postclock[w] < self._postclock[v]))
    
    def _isNonTreeEdge(self, v, w):
        return (v,w) in self._getNonTreeEdges()
    
    def _getNonTreeEdges(self):
        return self._backedges + self._forwardedges + self._crossedges
    
    def isConnected(self, v, w):
        return self.hasPath(v, w) and self.hasPath(w, v)

class DFSDirectedSCC: # strongly connected components
                      # O(V+E)
    def __init__(self, graph):
        self.graph = graph
        self._reversegraph = DFSDirectedSCC._getReverseGraph(graph)
        self._visited = [False] * graph.numvertices
        self._tsrg = DFSDirectedSCC._getTopologicalSort(self._reversegraph)
        self._sccnum = [None] * graph.numvertices
        self._scc = 0
        for v in self._tsrg:
            if not self._visited[v]:
                self._explore(v)
                self._scc += 1
    
    def getSCC(self):
        result = [[] for _ in range(self._scc)]
        for v in self._tsrg:
            result[self._sccnum[v]].append(v)
        return result
    
    def getNumSCC(self):
        return self._scc
    
    def _explore(self, v):
        self._visited[v] = True
        self._sccnum[v] = self._scc
        for w in self.graph.neighbors(v):
            if not self._visited[w]:
                self._explore(w)
    
    def _getReverseGraph(graph):
        R = DirectedGraph(graph.numvertices)
        R._entries, R._exits = graph._exits, graph._entries
        return R
    
    def _getTopologicalSort(graph):
        return DFSDirected(graph).getTopologicalSort()
        

## Python ##
def has_cycle_directed_dfs_stack(V, E):
    # marked[i] = 0 if V[i] not visited, 1 if on explore stack, 2 if off stack
    # cycle iff descendant on stack
    marked = [0]*(len(V))
    
    def explore(i):
        marked[i] = 1
        for j in E[i]:
            if marked[j] == 1:
                return True
            if not marked[j]:
                if explore(j):
                    return True
        marked[i] = 2
        return False
    
    return any(explore(i) for i in range(len(V)) if not marked[i])

def has_cycle_undirected_dfs_stack(V, E):
    visited = [False]*(len(V))
    
    def explore(i):
        visited[i] = True
        for j in E[i]:
            if visited[j]:
                return True
            if explore(j):
                return True
        return False
    
    return any(explore(i) for i in range(len(V)) if not visited[i])
    
def bipartite_undirected(V, E):
    color = [0]*(len(V))
    
    def explore(i, c):
        color[i] = c
        d = (c+1)%2
        for j in E[i]:
            if color[j] == color[i]:
                return True
            if not color[j]:
                if explore(j, d):
                    return True
        return False
    
    return any(explore(i) for i in range(len(V)) if not color[i])
    
def pouring_water():
    """Containers of size 4, 7, 10.  Start 4, 7 full.  End 2 y z or x 2 z.  Pour until container is full or other is empty."""
    size = (4, 7, 10)
    source = (4, 7, 0)
    def target(s):
        return s[0] == 2 or s[1] == 2
    visited = set()
    prev = {source:None}
    
    P = [(i,j) for i in range(3) for j in range(3) if i!=j]
    def pour(s, x, i, j):
        t = list(s)
        t[i] -= x
        t[j] += x
        return tuple(t)
        
    def amount(s, i, j):
        return min(s[i], size[j] - s[j])
    
    def neighbors(s):
        result = []
        for i,j in P:
            a = amount(s, i, j)
            if a:
                result.append(pour(s, x, i, j))
        return result
    
    def path(s):
        result = deque()
        while s:
            deque.appendleft(s)
            s = prev[s]
        return result
    
    def explore(s):
        if target(s):
            return path(s)
        visited.add(s)
        for t in neighbors(s):
            if not visited(t):
                explore(t)
    
    return explore(s)
    
from math import inf

def binary_tree_max_descendant(V, E):
    # V[i] = value at vertex i
    # result M
    visited = [False]*len(V)
    M = [-inf]*len(V)
    def explore(i):
        visited[i] = True
        for j in E[i]:
            if not visited[j]:
                M[i] = max(M[i], explore(j))
        return max(M[i], V[i])
    explore(0)
    return M