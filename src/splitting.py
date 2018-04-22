from math import log

GAIN = True
GAIN_RATIO = False

def countAttributeValues(D,a):
   counts = {}
   total = 0
   for datum in D:
      currentValue = datum[0][a]
      if currentValue not in counts:
         currentValue = 1
      else:
         currentValue += 1
      total += 1
   return counts, total

def countClassAttributeValues(D):
   counts = {}
   total = 0
   for datum in D:
      if datum[1] not in counts:
         counts[datum[1]] = 1
      else:
         counts[datum[1]] += 1
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
      d[curr] += datum
   return d

'''entropy
   D : the training dataset
   A : an attribute to filter by
'''

def entropy(D, a=None):
   sm = 0
   counts,total = countClassAttributeValues(D)
   if(a):
      splitted = splitDataSet(D,a)
      for lst in list(splitted.values()):
         sm += (float(len(lst))/total)*entropy(lst)
   else:
      k = len(counts)
      for i in range(1,k):
         pr = float(counts[k])/total
         sm -= pr*log(pr,2)
   return sm

def gain(D,a):
   return entropy(D) - entropy(D,a)

def gainRatio(D,a):
   num = gain(D,a)
   denom = 0
   splitted = splitDataSet(D,a)
   for lst in list(splitted.values()):
      pr = float(len(lst))/len(D)
      denom -= pr*log(pr,2)
   return float(num)/denom

def findMax(dct):
   maxKey = ""
   maxVal = -1
   for key in dct:
      if dct[key] > maxVal:
         maxVal = dct[key]
         maxKey = key
   return maxKey

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
   for a,attr in enumerate(A):
      attr = attr[0]
      print(attr)
      gains[attr] = gain(D,a)
      gainRatios[attr] = gainRatio(D,a)
   best = findMax(gains) if isgain else findMax(gainRatios)
   if gains[best] > threshold:
      return best
   else:
      return Null
