import argparse 
import xml.etree.ElementTree as ET
import pdb
import csv
import random

def parse_schema(filename):
    xml = ET.parse(args.schema_file)
    root = xml.getroot()
    schema = {}
    for var in root:
        schema[var.attrib['name']] = [ v.attrib for v in var ]
    return schema

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
            for i in range(0, len(row)):
                if attributes[i][1] < 0 or i == class_col:
                    row[i] = None
            row = [n for n in row if n != None]
            data.append((tuple(row), k))
        attributes = [a for a in attributes if a[1] >= 0 and a[0] != klass]
    return data, attributes, klass

class Node:
    NODE_TYPE = ['attribute', 'class']

    def __init__(self, node_type, label=None):
        if node_type not in self.NODE_TYPE:
            raise "'{}' is not a valid node type".format(node_type)
        self.node_type = node_type
        self.label = label
        self.children = {}

    def to_s(self, edge, indent):
        return " " * 2 * indent + "({}) {} {}".format(edge, self.node_type, self.label)


def print_tree(node, edge, indent):
    print(node.to_s(edge, indent))
    for e, n in node.children.items():
        print_tree(n, e, indent + 1)

def generate(D, A, threshold):
    klass = D[0][1]
    if all(d[0][1] == klass for d in D):
        return Node("class", klass)
    elif all(a is None for a in A):
        # TODO picks first klass in D, should find most frequent
        return Node("class", klass)
    else:
        # TODO selects random attribute, should find correct one
        selected_idx = random.randint(0, len(A) - 1)
        while not A[selected_idx]:
            selected_idx = random.randint(0, len(A) - 1)

        selected = A[selected_idx]
        A[selected_idx] = None
        node = Node("attribute", selected[0])
        for e in range(1, selected[1] + 1):
            node.children[e] = generate(D, list(A), threshold)
        return node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    args = parser.parse_args()
 
    schema = parse_schema(args.schema_file)
    data, attributes, klass = parse_data(args.data_file)

    root = generate(data, list(attributes), 0.0)
    print_tree(root, None, 0)
