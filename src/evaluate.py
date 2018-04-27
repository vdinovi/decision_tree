


def parse_inputs(filename):








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    args = parser.parse_args()


