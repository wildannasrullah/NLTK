from __future__ import print_function
from nltk.metrics import *
reference = 'DET NN VB DET JJ NN NN IN DET NN'.split()
test    = 'DET VB VB DET NN NN NN IN DET NN'.split()
print(accuracy(reference, test))
reference_set = set(reference)
test_set = set(test)
precision(reference_set, test_set)

print(recall(reference_set, test_set))

print(f_measure(reference_set, test_set))
