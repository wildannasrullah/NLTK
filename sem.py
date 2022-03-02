import nltk
from nltk.sem import Valuation, Model
from nltk.sem import *
v = [('adam', 'b1'), ('betty', 'g1'), ('fido', 'd1'),
    ('girl', set(['g1', 'g2'])), ('boy', set(['b1', 'b2'])),
    ('dog', set(['d1'])),
    ('love', set([('b1', 'g1'), ('b2', 'g2'), ('g1', 'b1'), ('g2', 'b1')]))]
val = Valuation(v)
dom = val.domain
m = Model(dom, val)
print(m)

