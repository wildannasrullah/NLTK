import nltk
from nltk import Nonterminal, nonterminals, Production, CFG
from nltk.parse import RecursiveDescentParser
from nltk.parse import ShiftReduceParser
nt1 = Nonterminal('NP')
nt2 = Nonterminal('VP')
nt1.symbol()
nt1 == Nonterminal('NP')

nt1 == nt2

S, NP, VP, PP = nonterminals('S, NP, VP, PP')
N, V, P, DT = nonterminals('N, V, P, DT')
prod1 = Production(S, [NP, VP])
prod2 = Production(NP, [DT, NP])
prod1.lhs()

prod1.rhs()

prod1 == Production(S, [NP, VP])

prod1 == prod2

grammar = CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> 'the' N | N PP | 'the' N PP
    VP -> V NP | V PP | V NP PP
    N -> 'cat'
    N -> 'dog'
    N -> 'rug'
    V -> 'chased'
    V -> 'sat'
    P -> 'in'
    P -> 'on'
""")
rd = RecursiveDescentParser(grammar)
sentence1 = 'the cat chased the dog'.split()
sentence2 = 'the cat chased the dog on the rug'.split()
for t in rd.parse(sentence1):
    print(t)
for t in rd.parse(sentence2):
    print(t)

nltk.parse.chart.demo(2, print_times=False, trace=1,
    sent='I saw a dog', numparses=1)
