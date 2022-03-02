import nltk

from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import abc

# sample text
sample = abc.raw("rural.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])
