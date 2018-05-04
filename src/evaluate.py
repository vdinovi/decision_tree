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


##############
#   Part 2   #
##############
def classify_set(data, attrs, node):
    total = 0
    success = 0
    failure = 0
    for d in data:
        if classify(d[0], attrs, node) == d[1]:
            success += 1
        else:
            failure += 1
        total += 1
    print("classified: {}".format(total))
    print("successes:  {}".format(success))
    print("failures:   {}".format(failure))
    print("accuracy:   {}".format(float(success) / failure))
    return (total, success, failure)

##############
#   Part 3   #
##############
def eval_f_measure(confusion_mat, beta):
    true_pos = confusion_mat[0][0]
    false_pos = confusion_mat[1][0]
    false_neg = confusion_mat[0][1]
    precision = float(true_pos) / ((true_pos + false_pos) or 0.00001)
    recall = float(true_pos) / ((true_pos + false_neg) or 0.00001)
    f_measure = (1 + beta * beta) * precision * recall / (beta * beta) * ((precision + recall) or 0.00001)
    return (precision, recall, f_measure)

def confusion(mat, actual, expected):
    mat[0][0] += sum([1 for i,_  in enumerate(actual) if actual[i] == target and expected[i] == target])
    mat[0][1] += sum([1 for i,_  in enumerate(actual) if actual[i] == target and expected[i] != target])
    mat[1][0] += sum([1 for i,_  in enumerate(actual) if actual[i] != target and expected[i] == target])
    mat[1][1] += sum([1 for i,_  in enumerate(actual) if actual[i] != target and expected[i] != target])


def accuracy(actual, expected):
    return sum([1 for i,_ in enumerate(actual) if actual[i] == expected[i]]) / float(len(actual))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cross_validate(data, attributes, target, threshold, gain_ratio, beta):
    num_folds = 10
    attrs = [a[0] for a in attributes]
    partitions = [chunk for chunk in chunks(data, num_folds)]
    confusion_mat = [[0, 0], [0, 0]] #target (row) x non-target (col)
    accuracies = []
    for i in range(0, num_folds):
        train_set = [x for part in partitions[0:i] + partitions[i+1:num_folds] for x in part]
        test_set = partitions[i]
        root = generate(train_set, attributes, threshold, gain_ratio)
        actual = [classify(d[0], attrs, root) for d in test_set]
        expected = [d[1] for d in test_set]
        confusion(confusion_mat, actual, expected)
        accuracies.append(accuracy(actual, expected))
    avg_accuracy = sum(accuracies) / float(len(accuracies))
    overall_accuracy = float(confusion_mat[0][0] + confusion_mat[1][1]) / (confusion_mat[0][0] + confusion_mat[0][1] + confusion_mat[1][0] + confusion_mat[1][1])
    precision, recall, f_measure = eval_f_measure(confusion_mat, beta)
    stats =  {
        "confusion_mat": confusion_mat,
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "overall_accuracy": overall_accuracy,
        "average_accuracy": avg_accuracy
    }
    return stats

def print_stats(stats, params):
    print("-------------- Stats --------------")
    print("Gain Ratio: {}".format(params["gain_ratio"]))
    print("Threshold: {}".format(params["threshold"]))
    print("Beta: {}".format(params["beta"]))
    print("Confusion Matrix:")
    print("  |TP={}|FN={}|".format(stats["confusion_mat"][0][0], stats["confusion_mat"][0][1]))
    print("  |FP={}|TN={}|".format(stats["confusion_mat"][1][0], stats["confusion_mat"][1][1]))
    print("Precision:{:03f}".format(stats["precision"]))
    print("Recall:{:03f}".format(stats["recall"]))
    print("F-measure: {:03f}".format(stats["f_measure"]))
    print("Overall Accuracy: {:03f}".format(stats["overall_accuracy"]))
    print("Average Accuracy: {:03f}".format(stats["average_accuracy"]))
    print("----------------------------------")

def write_stats(stats, params, outfile):
    with open(outfile, "a") as file:
        file.write("-------------- Stats --------------\n")
        file.write("Gain Ratio: {}\n".format(params["gain_ratio"]))
        file.write("Threshold: {}\n".format(params["threshold"]))
        file.write("Beta: {}\n".format(params["beta"]))
        file.write("Confusion Matrix:\n")
        file.write("  |TP={}|FN={}|\n".format(stats["confusion_mat"][0][0], stats["confusion_mat"][0][1]))
        file.write("  |FP={}|TN={}|\n".format(stats["confusion_mat"][1][0], stats["confusion_mat"][1][1]))
        file.write("Precision:{:03f}\n".format(stats["precision"]))
        file.write("Recall:{:03f}\n".format(stats["recall"]))
        file.write("F-measure: {:03f}\n".format(stats["f_measure"]))
        file.write("Overall Accuracy: {:03f}\n".format(stats["overall_accuracy"]))
        file.write("Average Accuracy: {:03f}\n".format(stats["avg_accuracy"]))
        file.write("----------------------------------\n")


def plot(max_thresh, min_thresh, data, attributes, target):
    import matplotlib.pyplot as plt
    dt = 0.0005
    thresholds = []
    accuracy = []
    f_measure = []
    thresh = float(max_thresh)
    # With Gain 
    with open("c45_gain.txt", "w") as file:
        while thresh >= min_thresh:
            stats = cross_validate(data, attributes[:], target, thresh, False)
            thresholds.append(thresh)
            accuracy.append(stats["average_accuracy"])
            f_measure.append(stats["f_measure"])
            file.write("({:4f}) avg_accuracy={:4f}, f_measure:{:4f}\n".format(thresh, stats["average_accuracy"], stats["f_measure"]))
            thresh -= dt
    plt.clf()
    plt.subplot(211)
    plt.xlabel("threshold")
    plt.ylabel("average accuracy")
    plt.plot(thresholds, accuracy)
    plt.title("C45 using Information Gain")
    plt.subplot(212)
    plt.xlabel("threshold")
    plt.ylabel("f-measure")
    plt.plot(thresholds, f_measure)
    plt.savefig("c45_gain.png")

    # With Gain Ratio
    thresholds = []
    accuracy = []
    f_measure = []
    thresh = float(max_thresh)
    with open("c45_gain_ratio.txt", "w") as file:
        while thresh >= min_thresh:
            stats = cross_validate(data, attributes[:], target, thresh, True)
            thresholds.append(thresh)
            accuracy.append(stats["average_accuracy"])
            f_measure.append(stats["f_measure"])
            file.write("({:4f}) avg_accuracy={:4f}, f_measure:{:4f}\n".format(thresh, stats["average_accuracy"], stats["f_measure"]))
            thresh -= dt
    plt.clf()
    plt.subplot(211)
    plt.xlabel("threshold")
    plt.ylabel("average accuracy")
    plt.plot(thresholds, accuracy)
    plt.title("C45 using Information Gain Ratio")
    plt.subplot(212)
    plt.xlabel("threshold")
    plt.ylabel("f-measure")
    plt.plot(thresholds, f_measure)
    plt.savefig("c45_gain_ratio.png")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("schema_file", help="xml file containing names")
    parser.add_argument("data_file", help="csv file containing numeric data")
    parser.add_argument("--gratio", action='store_true', help="specify if information gain ratio is to be used, else regular information gain")
    parser.add_argument("--threshold", help="specify threshold, else hardcoded value will be used")
    parser.add_argument("--beta", help="specify beta for use in the f-measure, else 1 will be used (>1 favors recall, <1 favors precision)")
    parser.add_argument("--plot", action='store_true', help="specify to plot thresholds/gain ratio")
    args = parser.parse_args()

    params = {}
    params["gain_ratio"] = args.gratio or False
    if args.threshold:
        params["threshold"] = float(args.threshold)
    else:
        params["threshold"] = 0.01
    if args.beta:
        params["beta"] = float(args.beta)
    else:
        params["beta"] = 1

    schema = parse_schema(args.schema_file)
    data, attributes, category = parse_data(args.data_file)
    target = int(list(filter(lambda x: x["name"] == "Obama", schema[category]))[0]["type"])
    filename = "c45eval_{}.png".format("gain_ratio" if params["gain_ratio"] else "gain")
    if args.plot:
        plot(0.2, 0.00, data, attributes, target)
    else:
        stats = cross_validate(data, attributes, target, params["threshold"], params["gain_ratio"], params["beta"])
        print_stats(stats, params)

