import argparse
import xml.etree.ElementTree as ET
import pdb
import csv
import random
from splitting import selectSplittingAttribute
from tree import Node, print_tree, generate
from pprint import pprint
import matplotlib.pyplot as plt

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

def classify(datum, attrs, node):
    if node.node_type == "class":
        return node.label
    else:
        edge = datum[attrs.index(node.label)]
        if edge not in node.children:
            # value for this attribute was not seen by our decision tree
            # -1 represents failure to classify this datum
            return -1
        else:
            return classify(datum, attrs, node.children[edge])

# use simple accuracy for now
def evaluate(actual, expected):
    return sum([1 for i,_  in enumerate(actual) if actual[i] != expected[i]])

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cross_validate(data, attributes, threshold, gain_ratio):
    num_folds = 10
    scores = []
    attrs = [a[0] for a in attributes]
    partitions = [chunk for chunk in chunks(data, num_folds)]
    for i in range(0, num_folds):
        train_set = [x for part in partitions[0:i] + partitions[i+1:num_folds] for x in part]
        test_set = partitions[i]
        root = generate(train_set, attributes, threshold, gain_ratio)
        result = [classify(d[0], attrs, root) for d in test_set]
        scores.append(evaluate(result, [d[1] for d in test_set]))
    return sum(scores) / float(len(scores))

def plot(filename, max_thresh, min_thresh, gain_ratio, data, attributes):
    dt = 0.01
    thresholds = []
    results = []
    thresh = float(max_thresh)
    while thresh >= min_thresh:
        thresholds.append(thresh)
        results.append(cross_validate(data, attributes[:], thresh, gain_ratio))
        thresh -= dt
    gain_label = "Gain Ratio" if gain_ratio else "Gain"
    plt.clf()
    plt.xlabel("threshold")
    plt.ylabel("score")
    plt.text(0.5, 0.5, "Method: {}".format(gain_label), fontsize=12)
    plt.plot(thresholds, results)
    plt.savefig(filename)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    parser.add_argument("--gratio", action='store_true', help="specify if information gain ratio is to be used, else regular information gain")
    parser.add_argument("--threshold", help="specify threshold, else hardcoded value will be used")
    args = parser.parse_args()

    gain_ratio = args.gratio or False
    if args.threshold:
        threshold = float(args.threshold)
    else:
        threshold = 0.1

    schema = parse_schema(args.schema_file)
    data, attributes, category = parse_data(args.data_file)
    filename = "c45eval_{}.png".format("gain_ratio" if gain_ratio else "gain")
    plot(filename, 0.4, 0.00, gain_ratio, data, attributes)

