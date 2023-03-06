class DFG:

    class Node:
        def __init__(self, event_type):
            self.event_type = event_type
            self.frequency = 1

        def increase(self):
            self.frequency += 1

    class Edge:
        def __init__(self, src, tgt):
            self.src = src
            self.tgt = tgt
            self.frequency = 1

        def increase(self):
            self.frequency += 1


    def __init__(self):
        self.nodes = None
        self.edges = None
        self.loops = None
        self.reachability = None

    def update(self, row, row_pre):
        node = self.Node(row['eventType'])
        src = self.Node(row_pre['eventType'])
        edge = self.Edge(src, node)

        if self.nodes.contains(node):
            self.nodes.get(node).increase()
            if self.edges.contains(edge):
                self.edges.get(edge).increase()
            else:
                self.edges.add(edge)
        else:
            self.nodes.add(node)
            self.edges.add(edge)
            self