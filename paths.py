

class BFS: # unweighted, directed or undirected
           # O(V+E)

    def __init__(self, G, source):
        self.graph = G
        self.dist = [float('inf')]*G.numvertices
        self.prev = [None]*G.numvertices
        self.queue = deque([source])
        self.dist[source] = 0
        self.explore()
    
    def explore(self):
        while self.queue:
            v = self.queue.popleft()
            for w in self.graph.adj(v):
                if self.dist[w] == float('inf'):
                    self.dist[w] = v+1
                    self.prev[w] = v
                    self.queue.append(w)
    
    def get_shortest_dist(self, v):
        return self.dist[v]
    
    def get_shortest_path(self, v):
        result = []
        while v != None:
            result.append(v)
            v = self.prev[v]
        result.reverse()
        return result

class Dijkstra: # nonnegative weighted, directed or undirected, single source
                # O((V+E)logV) = V*deletemin + V*insert + E*change_priority
                # if E > V**2/logV, use array instead of PQ: O(V**2) = V*V + V*1 + E*1

    def __init__(self, G, source):
        self.graph = G
        self.dist = [float('inf')]*G.numvertices
        self.dist[source] = 0
        self.prev = [None]*G.numvertices
        self.queue = MinPQ([(self.dist[v], v) for v in range(G.numvertices)])
        self.explore()
    
    def explore(self):
        while self.queue:
            d, v = self.queue.pop()
            for w in self.graph.adj(v):
                if d + self.graph.weight(v, w) < self.dist[w]:
                    # update(v, w)
                    self.dist[w] = d + self.graph.weight(v, w)
                    self.prev[w] = v
                    self.queue.change_priority(w, self.dist[w])

class Dijkstra: # nonnegative weighted, directed or undirected, single source
                # Growing PQ

    def __init__(self, G, source):
        self.graph = G
        self.dist = [float('inf')]*G.numvertices
        self.dist[source] = 0
        self.prev = [None]*G.numvertices
        self.queue = MinPQ([(0, source)])
        self.explore()
    
    def explore(self):
        while self.queue:
            d, v = self.queue.pop()
            for w in self.graph.adj(v):
                if d + self.graph.weight(v, w) < self.dist[w]:
                    # update(v, w)
                    self.dist[w] = d + self.graph.weight(v, w)
                    self.prev[w] = v
                    if w in self.queue:
                        self.queue.change_priority(w, self.dist[w])
                    else:
                        self.queue.add(w, self.dist[w])

class BellmanFord: # directed weighted, possibly negative (no negative cycles), single source

    def __init__(self, G, source):
        self.numvertices = G.numvertices
        self.edges = G.get_edges()
        self.dist = [float('inf')]*G.numvertices
        self.dist[source] = 0
        self.prev = [None]*G.numvertices
        self.explore()
    
    def explore(self):
        for updateround in range(self.numvertices-1):
            anyupdates = self.updates()
            if not anyupdates:
                break
        if updateround == self.numvertices-1:
            # check for negative cycles
            anyupdates = self.updates()
            if anyupdates:
                raise NegativeCycleException('Shortest path is ill-defined with negative cycles.')
    
    def updates(self):
        result = False
        for e in self.edges:
            u, v, d = e.source, e.target, e.weight
            if dist[u] + d < self.dist[v]:
                # update(u, v)
                self.dist[v] = dist[u] + d
                self.prev[v] = u
                result = True
        return result
            

## Exercises ##

# num shortest paths between source and target
# undirected, unit length graph

def num_shortest_paths(V, E, source, target):
    dist = [inf]*len(V)
    dist[source] = 0
    numto = [0]*len(V)
    numto[source] = 1
    queue = deque([source])
    while queue and dist[queue[0]] < dist[target]:
        u = queue.popleft()
        for v in E[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                numto[v] = numto[u]
            elif dist[v] == dist[u] + 1:
                numto[v] += numto[u]
    return numto[target]

def exchange_rates(R, source):
    cost = [inf]*len(R)
    cost[source] = 1
    prev = [None]*len(R)
    
    def update(R):
        result = False
        for i in range(len(R)):
            for j in range(len(R)):
                if cost[i]*R[i][j] < cost[j]:
                    cost[j] = cost[i]*R[i][j]
                    prev[j] = i
                    result = True
        return result
    
    i = 0
    while i < len(R)-1:
       if not update(R):
           break
    arbitrage = False
    if i == len(R)-1:
        if update(R):
            aribitrage = True
    return cost, arbitrage
    

class ArrayHeap:
    
    def __init__(self, nary=2, array=None):
        self.nary = nary
        if not array:
            self.array = []
        else:
            self.array = array
        self._heapify(array)
    
    def _heapify(self):
        pass
    

