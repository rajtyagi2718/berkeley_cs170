# Primal problem
# objective: max(c*x)
# constraints: Ax <= b
#              x >= 0
# Dual problem
# objective: min(y*b)
# constraints: y*A >= c*
#              y >= 0
# If feasible solution exists, the optimal solution is the pair x_hat, y_hat
# such that c*x_hat = y_hat b.
# Simplex method
# Find the vertices of the convex polygon inscribed by constraints. Hill climb
# vertices if neighboring vertex has better solution to objective function.
# Local extrema will be global extrema since region is convex.


class MaximumFlow:
    """Given capacity graph (positive weighted directed), find maximum flow
       graph from source to target. In flow, for each vertex, sum of in_edge
       weights equals sum of out_edge weights. All weights positive and less
       than capacity."""
    
    def __init__(self, capacity, source, target):
        assert self.is_source(source)
        assert self.is_sink(target)
        self.source = source
        self.target = target
        self.num_vertices = capacity.num_vertices
        self.flow = WeightedDirectedGraph(self.num_vertices)
        self.residual = capacity
        self.explore()
        self.size = sum(edge.weight for edge in
                        self.flow.get_out_edges(self.source))
    
    def is_source(self, vertex):
        return not self.graph.get_in_edges(vertex)
    
    def is_sink(self, vertex):
        return not self.graph.get_out_edges(vertex)
    
    def explore(self):
        path = self.dfs_get_path(self.residual, self.source, self.target)
        if not path:
            return
        for edge in path:
            self.increment_edge(self.flow, edge.source, edge.target)
            self.decrement_edge(self.residual, edge.source, edge.target)
            self.increment_edge(self.residual, edge.target, edge.source)
        return self.explore()
    
    @staticmethod
    def dfs_get_path(graph, source, target):
        """Depth first search for path from source to target."""
        marked = [False]*graph.num_vertices
        prev = [None]*graph.num_vertices
        
        def explore(vertex):
            marked[vertex] = True
            if vertex == target:
                return
            for adj in graph.get_out_edges(vertex):
                if not marked[adj.target]:
                    prev[adj] = adj
                    explore(adj)
        
        path = []
        edge = prev[target]
        while edge:
            path.append(edge)
            edge = prev[edge.source]
        return path[::-1]
        
    @staticmethod
    def increment_edge(graph, source, target):
        if graph.has_edge(source, target):
            curr_weight = graph.get_weight(source, target)
            graph.change_weight(source, target, curr_weight+1)
        else:
            graph.add_edge(source, target, 1)
    
    @staticmethod
    def decrement_edge(graph, source, target):
        assert graph.has_edge(source, target):
        curr_weight = graph.get_weight(source, target)
        if curr_weight == 1:
            graph.remove_edge(source, target)
        else:
            graph.change_weight(source, target, curr_weight-1)
            
    def get_flow(self):
        return self.flow
    
    def get_size(self):
        return self.size