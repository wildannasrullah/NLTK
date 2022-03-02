import nltk
import random
from nltk.corpus import conll2000
from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import ChunkParserI, TrigramTagger

chunked_sentence = conll2000.chunked_sents()[0]
iob_tagged = tree2conlltags(chunked_sentence)
print (iob_tagged)
 
# [(u'Confidence', u'NN', u'B-NP'), (u'in', u'IN', u'B-PP'), (u'the', u'DT', u'B-NP'), (u'pound', u'NN', u'I-NP'), (u'is', u'VBZ', u'B-VP'), (u'widely', u'RB', u'I-VP'), (u'expected', u'VBN', u'I-VP'), (u'to', u'TO', u'I-VP'), (u'take', u'VB', u'I-VP'), (u'another', u'DT', u'B-NP'), (u'sharp', u'JJ', u'I-NP'), (u'dive', u'NN', u'I-NP'), (u'if', u'IN', u'O'), (u'trade', u'NN', u'B-NP'), (u'figures', u'NNS', u'I-NP'), (u'for', u'IN', u'B-PP'), (u'September', u'NNP', u'B-NP'), (u',', u',', u'O'), (u'due', u'JJ', u'O'), (u'for', u'IN', u'B-PP'), (u'release', u'NN', u'B-NP'), (u'tomorrow', u'NN', u'B-NP'), (u',', u',', u'O'), (u'fail', u'VB', u'B-VP'), (u'to', u'TO', u'I-VP'), (u'show', u'VB', u'I-VP'), (u'a', u'DT', u'B-NP'), (u'substantial', u'JJ', u'I-NP'), (u'improvement', u'NN', u'I-NP'), (u'from', u'IN', u'B-PP'), (u'July', u'NNP', u'B-NP'), (u'and', u'CC', u'I-NP'), (u'August', u'NNP', u'I-NP'), (u"'s", u'POS', u'B-NP'), (u'near-record', u'JJ', u'I-NP'), (u'deficits', u'NNS', u'I-NP'), (u'.', u'.', u'O')]
 
 
chunk_tree = conlltags2tree(iob_tagged)
print (chunk_tree)
 
shuffled_conll_sents = list(conll2000.chunked_sents())
random.shuffle(shuffled_conll_sents)
train_sents = shuffled_conll_sents[:int(len(shuffled_conll_sents) * 0.9)]
test_sents = shuffled_conll_sents[int(len(shuffled_conll_sents) * 0.9 + 1):]

class TrigramChunkParser(ChunkParserI):
    def __init__(self, train_sents):
        # Extract only the (POS-TAG, IOB-CHUNK-TAG) pairs
        train_data = [[(pos_tag, chunk_tag) for word, pos_tag, chunk_tag in tree2conlltags(sent)] 
                      for sent in train_sents]
 
        # Train a TrigramTagger
        self.tagger = TrigramTagger(train_data)
 
    def parse(self, sentence):
        pos_tags = [pos for word, pos in sentence]
 
        # Get the Chunk tags
        tagged_pos_tags = self.tagger.tag(pos_tags)
 
        # Assemble the (word, pos, chunk) triplets
        conlltags = [(word, pos_tag, chunk_tag) 
                     for ((word, pos_tag), (pos_tag, chunk_tag)) in zip(sentence, tagged_pos_tags)]
 
        # Transform to tree
        return conlltags2tree(conlltags)
 
 
trigram_chunker = TrigramChunkParser(train_sents)
print (trigram_chunker.evaluate(test_sents))
