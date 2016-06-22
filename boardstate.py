#rotates coordinates by transform
def coortransform(origin,transform,size=8):
    #return origin
    transform %= 4
    while transform < 0:
        transform += 4
    if transform == 1:
        return size-origin[1]-1, origin[0]
    if transform == 2:
        return size-origin[0]-1,size-origin[1]-1
    if transform == 3:
        return origin[1],size-origin[0]-1
    return origin
#checks whether all the coordinates in a board are clear
def allClear(self,coors):
        for coor in coors:
            if not self.occupant(coor) is None:
                return False
        return True
#a board occupied by legal pieces
class boardState:
    #transform dictates is where white starts
    #    2
    #    |
    #   +v+
    #3--> <--1
    #   +^+
    #    |
    #    0
    def __init__(self,whiteToken,blackToken,size=8):
        self.tokenW=whiteToken
        self.tokenB=blackToken
        self.data=[[None for _ in range(size)] for _ in range(size)]
        self.transform = 0
        self.size = size
        self.pieces = None
        self.prevTurn = None
    #get occupant of board
    def occupant(self,x,y=None,transformcoor=False):
        if y is None:
            y = x[1]
            x = x[0]
        if transformcoor:
            x,y = coortransform((x,y),self.transform,self.size)
        return self.data[x][y]
    #assign piece to spot on the board
    def assign(self,piece,x,y=None):
        if y is None:
            y = x[1]
            x = x[0]
        #x,y = coortransform((x,y),self.transform,self.size)
        self.data[x][y] = piece
    #rotates the board
    def reverse(self):
        return TransformBoardState(self.data,self.transform,self.size)
    #adds piece to the board at specified location
    def insertPiece(self,piece,location):
        piece.location = location
        self.assign(piece,location)
        if self.pieces is None:
            self.pieces = []
        self.pieces.append(piece)
    #advance the board's logic by one turn
    def advance(self):
        for p in self.pieces:
            p.advanceTime()
    #wrapper class representing a rotated board
class TransformBoardState:
    def __init__(self,source,transform,size):
        self.transform = transform
        self.size = size
        self.source = source
    def occupant(self,x,y=None,transformcoor=False):
        if y is None:
            y = x[1]
            x = x[0]
        if transformcoor:
            x,y = coortransform((x,y),self.transform,self.size)
        return self.source[x][y]