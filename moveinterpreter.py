from itertools import *
import copy
class boardChange:
    def __init__(self, kind, coordinates):
        self.coordinates = coordinates
        self.kind = kind
    def __repr__(self):
        return "{} at {}".format(self.kind,self.coordinates)


def Difference(this, other):
    if this is None:
        this = board(other.Size())
    if other is None:
        other = board(other.Size())
    for i in xrange(this.Size()):
        for j in xrange(this.Size()):
            if this[i, j] != other[i, j]:
                yield (this[i, j], other[i, j]), (i, j)


def Changes(this, other):
    appears = []
    disappears = []
    for kinds, coordinate in Difference(this,other):
        if not kinds[1] is None:
            appears.append(boardChange(kinds[1], coordinate))
        if not kinds[0] is None:
            disappears.append(boardChange(kinds[0], coordinate))
    return appears, disappears

class board:
    def __init__(self, size):
        self.data = [[None for x in range(size)] for x in range(size)]
    def Size(self):
        return len(self.data)
    def __getitem__(self, pos):
        row, col = pos
        return self.data[row][col]
    def __setitem__(self, pos , value):
        row, col = pos
        self.data[row][col] = value

class _noMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears)==0 and len(disappears)==0

class _regularMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears)==1 and len(disappears)==1 and appears[0].kind == disappears[0].kind

class _eatMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears)==1 and len(disappears)==2 and \
            disappears[1].kind != disappears[0].kind and \
            any(map(lambda x: x.coordinates == appears[0].coordinates, disappears))

class _irregularEatMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears)==1 and len(disappears)==2 and \
            disappears[1].kind != disappears[0].kind and \
            not any(map(lambda x: x.coordinates == appears[0].coordinates, disappears))

class _multiMoveMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears) == len(disappears) and all(imap(lambda x:x.kind==appears[0].kind,chain(appears,disappears)))

class _disappearMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears)==0 and len(disappears)!=0

class _appearMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return len(appears) != 0 and len(disappears) == 0

class _otherMoveType:
    def __init__(self):
        pass
    def isValid(self, appears, disappears):
        return True

class moveTypes:
    # accepted types
    def __init__(self):
        pass
    #no is for no changes
    no = _noMoveType()
    # regular is when one appears and disappears of the same type
    regular = _regularMoveType()
    # eat is when two tokens of different types disappear and one appears in place of either
    eat = _eatMoveType()
    # irregular eat is when two tokens of different types disappear and one appears in place of neither
    irregularEat = _irregularEatMoveType()
    # multimove is when two or more tokens disappear and two or more tokens appear, all of the same type
    multiMove = _multiMoveMoveType()
    # disappear is when tokens disappear and none appear (to use for promotion?)
    disappear = _disappearMoveType()
    # appear is when tokens appear but none disappear (usually only initial board set)
    appear = _appearMoveType()
    # means none of the above has occurred, usually an illegal move
    other = _otherMoveType()
    moveTypes = [no, regular, eat, irregularEat, multiMove, disappear, appear, other]

class move:
    def __init__(self, appears, disappears):
        self.type = ifilter(lambda x:x.isValid(appears,disappears),moveTypes.moveTypes).next()
        self.appears = appears
        self.disappears = disappears

class moveInterpreter:
    def __init__(self):
        self.prevs = [None]
    @property
    def prev(self):
        return self.prevs[-1]
    @prev.setter
    def prev(self, value):
        self.prevs.append(copy.deepcopy(value))
    def nextmove(self, currentBoard):
        prev = self.prev
        changes = Changes(prev, currentBoard)
        return move(changes[0],changes[1])
    def commit(self, currentBoard):
        self.prev = currentBoard
    def rollBack(self):
        self.prevs = self.prevs[:-1]