import argparse 
import xml.etree.ElementTree as ET
import pdb

def parse_schema(filename):
    xml = ET.parse(args.schema_file)
    root = xml.getroot()
    schema = {}
    for var in root:
        schema[var.attrib] = {el.attrib for el in var}




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    args = parser.parse_args()
 

    parse_schema(args.schema_file)
