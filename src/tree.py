import json
from math import log
import pdb

class Node:
    NODE_TYPES = ['attribute', 'n_attribute', 'class']

    def __init__(self, node_type, label=None,alpha=None):
        if node_type not in self.NODE_TYPES:
            raise "'{}' is not a valid node type".format(node_type)
        self.node_type = node_type
        self.label = label
        self.alpha = alpha
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


def print_tree(node, edge, indent, attr, schema, classes):
    print(node.to_s(edge, indent, attr, schema, classes))
    for e, n in node.children.items():
        print_tree(n, e, indent + 1, schema[node.label], schema, classes)

def getEdgeName(schema,attribute,num):
   return schema[attribute][num-1]["name"]

def getFinalValue(schema,klass,num):
   arr = [x['name'] for x in schema[klass] if x['type'] == str(num)]
   return arr[0]

def format_for_json(n,schema,klass):
   d = {}
   if n.node_type == 'attribute':
      choices = {}
      d["name"] = n.label
      for key in n.children:
         choices[getEdgeName(schema,n.label,key)] = format_for_json(n.children[key],schema,klass)
      d["choices"] = choices
   elif n.node_type == 'n_attribute':
      choices = {}
      d["name"] = n.label
      for key in n.children:
         choices[getEdgeName(schema,n.label,key)] = format_for_json(n.children[key].schema,klass)
      d["choices"] = choices
   else: #node_type = class
      num = n.label
      d = getFinalValue(schema,klass,num)
   return d

def format_for_json_without_names(node):
    d = {}
    if node.node_type == 'attribute':
        choices = {}
        d["name"] = node.label
        for option, child in enumerate(node.children):
            choices[option] = format_for_json_without_names(child)
        d["choices"] = choices
    elif node.node_type == 'n_attribute':
        choices = {}
        d["name"] = node.label
        d["alpha"] = node.alpha
        for option, child in enumerate(node.children):
            choices[option] = format_for_json_without_names(child)
        d["choices"] = choices
    else:
        d = node.label
    return d

def to_json(node,filename,schema,klass):
   with open(filename,"w") as f:
      json.dump(format_for_json(node,schema,klass),f)

def to_json_nameless(node, filename):
    with open(filename, 'w') as f:
        json.dump(format_for_json_without_names(node), f)

def from_json(filename,schema,klass):
    pdb.set_trace()
    with open(filename,"r") as f:
        json_str = f.read()
        return json.loads(json_str)
        # make it back into a tree

def dominant_class(D):
    classes = {}
    for d in D:
        if d[1] in classes:
            classes[d[1]] = classes[d[1]] + 1
        else:
            classes[d[1]] = 1
    return max(classes)

GAIN = True
GAIN_RATIO = False

def countAttributeValues(D,a):
    counts = {}
    total = 0
    for datum in D:
        currentValue = datum[0][a]
        if currentValue not in counts:
            counts[currentValue] = 1
        else:
            counts[currentValue] += 1
        total += 1
    return counts, total

def countClassAttributeValues(D):
    counts = {}
    total = 0
    for datum in D:
        currentValue = datum[1]
        if currentValue not in counts:
            counts[currentValue] = 1
        else:
            counts[currentValue] += 1
        total+=1
    return counts,total

'''
   Splits data according to an attribute
   D : dataset
   a : attribute index to split on

'''
def splitDataSet(D,a):
    counts, total = countAttributeValues(D,a)
    d = dict((count,[]) for count in counts)
    for datum in D:
        curr = datum[0][a]
        d[curr].append(datum)
    return d

def splitDataSetNumeric(D,a,alpha):
    d = {1:[],2:[]}
    for datum in D:
        curr = datum[0][a]
        if(curr <= alpha):
            d[1].append(datum)
        else:
            d[2].append(datum)
    return d


'''entropy
   D : the training dataset
   A : an attribute to filter by
'''

def entropy(D, a=None, alpha=None):
    sm = 0
    counts,total = countClassAttributeValues(D)
    if(a):
        if(alpha):
            splitted = splitDataSetNumeric(D,a,alpha)
        else:
            splitted = splitDataSet(D,a)
        for lst in list(splitted.values()):
            sm += (float(len(lst))/total)*entropy(lst)
    else:
        k = len(counts)
        for i in range(1,k):
            pr = float(counts[k])/total
            sm -= pr*log(pr,2)
    return sm

def gain(D,a,alpha=None):
    return entropy(D) - entropy(D,a,alpha)

def gainRatio(D,a,alpha=None):
    num = gain(D,a,alpha)
    denom = 0
    if(alpha):
        splitted = splitDataSetNumeric(D,a,alpha)
    else:
        splitted = splitDataSet(D,a)
    for lst in list(splitted.values()):
        pr = float(len(lst))/len(D)
        denom -= pr*log(pr,2)
    if(denom == 0):
       denom = .0000001
    return float(num)/denom

def findMax(dct):
    maxKey = ""
    maxVal = -1
    for key in dct:
        if dct[key] > maxVal:
            maxVal = dct[key]
            maxKey = key
    return maxKey, maxVal

def getAttrIndex(A,a):
    for i,attr in enumerate(A):
        if (attr is not None) and attr[0] == a:
            return i
    return None

'''selectSplittingAttribute
A         : a list of attributes
D         : the training dataset
threshold : the lowest acceptable entropy to continue making a tree
isgain    : GAIN Ftrue) if you want to use Information Gain,
            GAIN_RATIO (false) if you want to use Information Gain Ratio.
'''
def selectSplittingAttribute(A,D,threshold,isgain=True):
    gains = {}
    gainRatios = {}
    for a, attr in enumerate(A):
        if(attr is not None):
            attr = attr[0]
            gains[attr] = gain(D,a)
            gainRatios[attr] = gainRatio(D,a)
    best,_ = findMax(gains) if isgain else findMax(gainRatios)
    if gains[best] > threshold:
        return best, getAttrIndex(A,best)
    else:
        return None, None

'''
Used for numeric attributes
'''

def bestGains(D,a):
    bestGain = -1
    bestGainRatio = -1
    bestGainAlpha = None
    bestGainRatioAlpha = None
    alphas = list(set([datum[0][a] for datum in D])) #these are the alphas to possibly split on
    for alpha in alphas:
        currGain = gain(D,a,alpha)
        currGainRatio = gainRatio(D,a,alpha)
        if currGain > bestGain:
            bestGain = currGain
            bestGainAlpha = alpha
        if currGainRatio > bestGainRatio:
            bestGainRatio = currGainRatio
            bestGainRatioAlpha = alpha
    return bestGain, bestGainAlpha, bestGainRatio, bestGainRatioAlpha

def selectSplittingAttributeNumerical(A,D,threshold,isgain=True):
   gains = {}
   gainRatios = {}
   gainAlphas = {}
   gainRatioAlphas = {}
   for a, attr in enumerate(A):
      if(attr is not None):
          attr = attr[0]
          gains[attr], gainAlphas[attr], gainRatios[attr], gainRatioAlphas[attr] = bestGains(D,a)
   best,bestKey = findMax(gains) if isgain else findMax(gainRatios)
   bestAlpha = gainAlphas[bestKey] if isgain else gainRatioAlphas[bestKey]
   if gains[best] > threshold:
      return best, gettAttrIndex(A,best),bestAlpha
   else:
      return None, None, None


def generate(D, A, threshold, gain_ratio):
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

