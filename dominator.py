class DominatorTree:
    def __init__(self, adj):
        self.adj_list = adj
        self.nodes = {}
        self.vertex = []

    class InfoNode:
        def __init__(self, node, parent):
            self.pred = []
            self.bucket = []
            self.node = node
            self.parent = parent
            self.dom = self

    def DFS(self, node, parent):
        if parent is not None:
            info = DominatorTree.InfoNode(node, self.nodes[parent])
        else:
            info = DominatorTree.InfoNode(node, None)
        info.dfsnum = len(self.vertex)
        info.semi = info.dfsnum
        info.label = info

        self.vertex.append(info)
        self.nodes[node] = info
        for succ in self.adj_list:
            if not self.nodes:
                if succ not in self.nodes:
                    self.DFS(succ, node)
                self.nodes[succ].pred.append(info)

    def COMPRESS(self, v):
        if v.ancestor is not None:
            self.COMPRESS(v.ancestor)
            if v.ancestor.label.semi < v.label.semi:
                v.label = v.ancestor.label
            v.ancestor = v.ancestor.ancestor

    def EVAL(self, v):
        if v.ancestor is None:
            return v
        else:
            self.COMPRESS(v)
            return v.label

    def analyse(self, root):
        self.DFS(root, None)
        for i in reversed(range(len(self.vertex))):
            w = self.vertex[i]

            for v in w.pred:
                u = self.EVAL(v)
                if u.semi < w.semi:
                    w.semi = u.semi
            self.vertex[w.semi].bucket.append(w)
            w.ancestor = w.parent

            if w.parent is not None:
                for v in w.parent.bucket:
                    u = self.EVAL(v)
                    if u.semi < v.semi:
                        v.dom = u
                    else:
                        v.dom = w.parent
                w.parent.bucket = []

        for i in range(len(self.vertex)):
            w = self.vertex[i]
            if w.dom != self.vertex[w.semi]:
                w.dom = w.dom.dom
        self.nodes[root].dom = None


if __name__ == '__main__':
    # Given Input
    vertices = ['copyCell', 'paste', 'editField', 'clickButton', 'clickLink', 'getCell', 'clickCheckbox', 'copy']
    adj = []
    edges = {'clickButton': ['clickLink', 'copyCell'], 'clickCheckbox': ['editField'], 'clickLink': ['copyCell', 'getCell', 'paste'], 'copyCell': ['clickLink', 'paste'], 'editField': ['clickButton', 'clickCheckbox', 'copyCell', 'editField', 'getCell', 'paste'], 'getCell': ['copyCell', 'getCell'], 'paste': ['copyCell', 'editField', 'paste']}
    for src in edges:
        for tgt in edges[src]:
            adj.append([vertices.index(src), vertices.index(tgt)])
    d = DominatorTree(adj)
    d.analyse(0)