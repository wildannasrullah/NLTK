import nltk
from nltk.probability import *

corpus = nltk.corpus.brown.tagged_sents(categories='adventure')[:500]
print(len(corpus))

from nltk.util import unique_list
tag_set = unique_list(tag for sent in corpus for (word,tag) in sent)
print(len(tag_set))

symbols = unique_list(word for sent in corpus for (word,tag) in sent)

print("Panjang Symbols :" ,len(symbols))

print("Panjang Tag Set : ", len(tag_set))

trainer = nltk.tag.HiddenMarkovModelTrainer(tag_set, symbols)

train_corpus = []
test_corpus = []
for i in range(len(corpus)):
    if i % 10:
         train_corpus += [corpus[i]]
    else:
        test_corpus += [corpus[i]]
print("Panjang Train Corpus : ", len(train_corpus))

print("Panjang Test Corpus : ",len(test_corpus))
def train_and_test(est):
    hmm = trainer.train_supervised(train_corpus, estimator=est)
    print('Evaluasi Train and Test Corpus : %.2f%%' % (100 * hmm.evaluate(test_corpus)))
mle = lambda fd, bins: MLEProbDist(fd)
train_and_test(mle)
train_and_test(LaplaceProbDist)
train_and_test(ELEProbDist)
