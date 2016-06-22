#testing file
#tests the move interperter
from moveinterpreter import board,moveInterpreter,moveTypes
from itertools import *
boardsize = 5
def loadBoards(source, size, noneToken ='0'):
    ret = None
    row = 0
    for line in source:
        if line.startswith('#'):
            continue
        if ret is None:
            ret = board(size)
        for token, col in izip(line,xrange(size)):
            if token == noneToken:
                token = None
            ret[row,col] = token
        row+=1
        if row == size:
            yield ret
            ret = None
            row = 0


interpreter = moveInterpreter()
for b, ind in izip(loadBoards(file("interpretertestboards.txt"),boardsize),count(1)):
    move = interpreter.nextmove(b)
    interpreter.commit(b)
    print '#{}- {}:'.format(ind,move.type.__class__.__name__)
    if len(move.appears) > 0:
        print '|Appears:'
        for a in move.appears:
            print '|{} at {}'.format(a.kind,a.coordinates)
    if len(move.disappears) > 0:
        print '|Disappears:'
        for a in move.disappears:
            print '|{} at {}'.format(a.kind, a.coordinates)
