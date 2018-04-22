import argparse 
import xml.etree.ElementTree as ET
import pdb
import csv
import random
from splitting import selectSplittingAttribute
from pprint import pprint

# Parse the schema XML file -> dictionary
def parse_schema(filename):
    xml = ET.parse(args.schema_file)
    root = xml.getroot()
    schema = {}
    for var in root:
        schema[var.attrib['name']] = [ v.attrib for v in var ]
    return schema

# Parse the data file into appropriate format
def parse_data(filename):
    data = []
    attributes = None
    klass = None
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        attrs = next(reader)
        counts = [int(n) for n in next(reader)]
        klass = next(reader)[0]

        attributes = [(attrs[i], counts[i]) for i in range(0, len(attrs))]
        class_col = attrs.index(klass)
        for row in reader:
            row = [int(n) for n in row]
            k = row[class_col]
            # strip out non-attribute columns and the class column from the data
            for i in range(0, len(row)):
                if attributes[i][1] < 0 or i == class_col:
                    row[i] = None
            row = [n for n in row if n != None]
            # store data elements in the form: (data_vector, class)
            data.append((tuple(row), k))
        # strip out non-attribute columns and the class column from the attributes
        attributes = [a for a in attributes if a[1] >= 0 and a[0] != klass]
    return data, attributes, klass

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

def generate(D, A, threshold):
    klass = D[0][1]
    if all(d[1] == klass for d in D):
        return Node("class", klass)
    elif all(a is None for a in A):
        klass = dominant_class(D)
        return Node("class", klass)
    else:
        selected_attr, selected_idx = selectSplittingAttribute(A,D,threshold)
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
                node.children[e] = generate(new_data, list(A), threshold)
        return node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    args = parser.parse_args()

    schema = parse_schema(args.schema_file)
    data, attributes, category = parse_data(args.data_file)

    root = generate(data, list(attributes), 0.0)
    print_tree(root, "", 0, {}, schema, schema[category])

