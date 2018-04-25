import argparse 
import xml.etree.ElementTree as ET
import pdb
import csv
import random
from splitting import selectSplittingAttribute
from tree import Node, print_tree, generate
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    args = parser.parse_args()

    schema = parse_schema(args.schema_file)
    data, attributes, category = parse_data(args.data_file)

    root = generate(data, list(attributes), 0.0)
    print_tree(root, "", 0, {}, schema, schema[category])

