import json
from splitting import selectSplittingAttribute

class Node:
    NODE_TYPES = ['attribute', 'class']

    def __init__(self, node_type, label=None):
        if node_type not in self.NODE_TYPES:
            raise "'{}' is not a valid node type".format(node_type)
        self.node_type = node_type
        self.label = label
        self.children = {}

    def to_s(self, edge, indent, attr, schema, classes):
        if edge:
            edge = "({}) ".format(attr[edge - 1]['name'])
        else:
            edge = ""
        if self.node_type == "class":
            label = classes[self.label - 1]['name']
        else:
            label = self.label
        return " " * 2 * indent + "{}{} {}".format(edge, self.node_type, label)

    def to_json(self):
        raise "NYI"

def print_tree(node, edge, indent, attr, schema, classes):
    print(node.to_s(edge, indent, attr, schema, classes))
    for e, n in node.children.items():
        print_tree(n, e, indent + 1, schema[node.label], schema, classes)

def to_dict(node):
   d = {}

   return d

def to_json(node):
   raise "NYI"

def dominant_class(D):
    classes = {}
    for d in D:
        if d[1] in classes:
            classes[d[1]] = classes[d[1]] + 1
        else:
            classes[d[1]] = 1
    return max(classes)

def generateForIris(D, threshold, gain_ratio):
    A = [('SepalLength', None), ('Sepal Width', None), ('Petal Length', None), ('Petal Width', None)]
    klass = D[0][1]
    if all(d[1] == klass for d in D):
        return Node("class", klass)
    elif all(a is None for a in A):
        klass = dominant_class(D)
        return Node("class", klass)
    else:
        selected_attr, selected_idx = selectSplittingAttribute(A,D,threshold, gain_ratio,)
        if not selected_idx:
            klass = dominant_class(D)
            return Node("class", klass)
        #selected_idx = random.randint(0, len(A) - 1)
        #while not A[selected_idx]:
        #    selected_idx = random.randint(0, len(A) - 1)

        # The selected attribute is set to None in the attribute list
        # this preserves the capability to parallel index into data vector
        selected = A[selected_idx]
        A[selected_idx] = None
        node = Node("attribute", selected[0])
        for e in range(1, selected[1] + 1):
            new_data = [d for d in D if d[0][selected_idx] == e]
            if new_data:
                node.children[e] = generate(new_data, list(A), threshold, gain_ratio)
        return node

def generate(D, A, threshold, gain_ratio,numeric=False):
    klass = D[0][1]
    if all(d[1] == klass for d in D):
        return Node("class", klass)
    elif all(a is None for a in A):
        klass = dominant_class(D)
        return Node("class", klass)
    else:
        selected_attr, selected_idx = selectSplittingAttribute(A,D,threshold, gain_ratio)
        if not selected_idx:
            klass = dominant_class(D)
            return Node("class", klass)
        #selected_idx = random.randint(0, len(A) - 1)
        #while not A[selected_idx]:
        #    selected_idx = random.randint(0, len(A) - 1)

        # The selected attribute is set to None in the attribute list
        # this preserves the capability to parallel index into data vector
        selected = A[selected_idx]
        A[selected_idx] = None
        node = Node("attribute", selected[0])
        for e in range(1, selected[1] + 1):
            new_data = [d for d in D if d[0][selected_idx] == e]
            if new_data:
                node.children[e] = generate(new_data, list(A), threshold, gain_ratio)
        return node

