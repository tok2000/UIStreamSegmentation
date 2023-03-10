def discover_scc(vertices, dfg):
    """
    Find the strongly connected components of a directed graph.
    Uses a recursive linear-time algorithm described by Tarjan [2]_ to find all
    strongly connected components of a directed graph.
    Parameters
    ----------
    vertices : iterable
        A sequence or other iterable of vertices.  Each vertex should be
        hashable.
    edges : mapping
        Dictionary (or mapping) that maps each vertex v to an iterable of the
        vertices w that are linked to v by a directed edge (v, w).
    Returns
    -------
    components : iterator
        An iterator that yields sets of vertices.  Each set produced gives the
        vertices of one strongly connected component.
    Raises
    ------
    RuntimeError
        If the graph is deep enough that the algorithm exceeds Python's
        recursion limit.
    Notes
    -----
    The algorithm has running time proportional to the total number of vertices
    and edges.  It's practical to use this algorithm on graphs with hundreds of
    thousands of vertices and edges.
    The algorithm is recursive.  Deep graphs may cause Python to exceed its
    recursion limit.
    `vertices` will be iterated over exactly once, and `edges[v]` will be
    iterated over exactly once for each vertex `v`.  `edges[v]` is permitted to
    specify the same vertex multiple times, and it's permissible for `edges[v]`
    to include `v` itself.  (In graph-theoretic terms, loops and multiple edges
    are permitted.)
    References
    ----------
    .. [1] Harold N. Gabow, "Path-based depth-first search for strong and
       biconnected components," Inf. Process. Lett. 74 (2000) 107--114.
    .. [2] Robert E. Tarjan, "Depth-first search and linear graph algorithms,"
       SIAM J.Comput. 1 (2) (1972) 146--160.
    """
    identified = set()
    stack = []
    index = {}
    lowlink = {}
    edges = {}

    def dfs(v):
        # Set the depth index for v to the smallest unused index
        index[v] = len(stack)
        stack.append(v)
        lowlink[v] = index[v]

        # Consider successors of v
        for w in edges[v]:
            if w not in index:
                # Successor w has not yet been visited; recurse on it
                yield from dfs(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w not in identified:
                # Successor w is in stack S and hence in the current SCC
                # If w is not on stack, then (v, w) is an edge pointing to an SCC already found and must be ignored
                lowlink[v] = min(lowlink[v], index[w])

        # If v is a root node, pop the stack and generate an SCC
        if lowlink[v] == index[v]:
            scc = set(stack[index[v]:])
            del stack[index[v]:]
            identified.update(scc)
            yield scc

    for edge in dfg:
        src = edge[0]
        tgt = edge[1]
        if src in edges:
            edges[src].append(tgt)
        else:
            edges[src] = [tgt]
    print("edges:")
    print(edges)

    for v in vertices:
        if v not in index:
            yield from dfs(v)

def get_scc_edges(dfg, scc):
    edges = []
    for e in dfg:
        if e[0] in scc and e[1] in scc:
            edges.append((e[0], e[1]))
    return edges



def discover_backedges(dfg, scc):
    for edge in get_scc_edges(dfg, scc):
        print(edge)
