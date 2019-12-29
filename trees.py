
class KruskalsMST: # O((V+E)lg*(V))
    
    def __init__(self, G):
        self.numvertices = G.numvertices
        self.tree_edges = []
        self._stack = sorted(G.edges, key=lambda x: -x.weight)
        self._ds = DisjointSet(range(G.numvertices))
        self._explore()
        
    def _explore(self):
        if len(self._stack) < self.numvertices:
            return None
        while len(self.tree_edges) < self.numvertices and self._stack:
            e = self._stack.pop()
            if not self._ds.is_connected(e.source, e.target):
                self._ds.connect(e.source, e.target)
                self.tree_edges.append(e)
        if len(self.tree_edges) < self.numvertices:
            return []
        return self.tree_edges
    

class PrimsMST:
    
    def __init__(self, G):
        self._dist = [float('inf')]*G.numvertices
        self._dist[0] = 0
        self.prev = [None]*G.numvertices
        self._pq = MinPQ((dist[v], v) for v in range(G.numvertices))
        self._adj = self.graph.adj
        self._explore()
    
    def _explore(self):
        while self._pq:
            d, s = self._pq.popmin()
            for e in self._adj(v):
                w, t = e.weight, e.target
                if w < dist[t]:
                    dist[t] = w
                    self._pq.change_priority(t, w)
                    prev[t] = s


class DisjointSet:
    
    pass

class HuffmanCode:
    
    pass

## exercises ##

def party(n, edges):
    adj = defaultdict([])
    for u,v in edges:
        adj[u].append(v)
        adj[v].append(u)
    change = True
    while change:
        change = False
        to_del = []
        for u in adj.keys():
            if len(adj[u]) < 5 or len(adj[u]) > len(adj.keys)-6:
                change = True
                to_del.append(u)
                for v in adj[u]:
                    adj[v].remove(u)
                    if not adj[v]:
                        to_del.append(v)
        for u in to_del:
            del adj[u]
    return len(adj)
            