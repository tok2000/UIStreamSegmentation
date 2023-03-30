# import the bitarray library
from bitarray import bitarray

# Declare bitarrays for each
# vertex to store the set of
# dominant vertices
b = [bitarray(100) for i in range(100)]

# Visited list to check if
# a vertex has been visited or not
vis = [0] * 100


def find_dominator(graph, parent, node):
    # If node is unvisited
    if vis[node] == 0:
        # Set all bits of b[pos]
        b[node].setall(True)

        # Update vis[node] to 1
        vis[node] = 1

    # Update b[node] with bitwise and
    # of parent's dominant vertices
    b[node] &= parent

    # Node is itself is a
    # dominant vertex of node
    b[node][node] = True

    # Traverse the neighbours of node
    for i in range(len(graph[node])):
        # Recursive function call to
        #  children nodes of node
        find_dominator(graph, b[node], graph[node][i])


def buildGraph(adj, E, V):
    # List of lists to store
    # the adjacency matrix
    graph = [[] for i in range(V + 1)]

    # Build the adjacency matrix
    for i in range(E):
        graph[adj[i][0]].append(adj[i][1])

    # Bitarray for node 0
    g = bitarray(100)
    g.setall(False)

    # Node 0 itself is a dominant
    # vertex of itself
    g[0] = True

    # Update visited of source
    # node as true
    vis[0] = 1

    # DFS from source vertex
    find_dominator(graph, g, 0)


def dominantVertices(vertices, E, adj):
    # Function call to build the graph
    # and dominant vertices
    V = len(vertices)
    buildGraph(adj, E, V)

    # Print set of dominating vertices
    for i in range(V):
        print(vertices[i], " -> ", end="")
        for j in range(V):
            if b[i][j]:
                print(vertices[j], end=" ")
        print()


# Driver Code

if __name__ == '__main__':
    # Given Input
    vertices = ['copyCell', 'paste', 'editField', 'clickButton', 'clickLink', 'getCell', 'clickCheckbox', 'copy']
    E = 0
    adj = []
    edges = {'clickButton': ['clickLink', 'copyCell'], 'clickCheckbox': ['editField'], 'clickLink': ['copyCell', 'getCell', 'paste'], 'copyCell': ['clickLink', 'paste'], 'editField': ['clickButton', 'clickCheckbox', 'copyCell', 'editField', 'getCell', 'paste'], 'getCell': ['copyCell', 'getCell'], 'paste': ['copyCell', 'editField', 'paste']}
    for src in edges:
        for tgt in edges[src]:
            adj.append([vertices.index(src), vertices.index(tgt)])
            E += 1
    print(adj)
    # Function Call
    dominantVertices(vertices, E, adj)
# this code is contributed by devendrasaluke