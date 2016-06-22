from boardstate import allClear
#returns the differnce btween two points
def locationdiff(origin,target):
    return target[0]-origin[0],target[1]-origin[1]
#returns the sign of a number
def sign(num):
    if num > 0:
        return 1
    if num < 0:
        return -1
    return 0
#brings one coordinate closer to another with 1 step
def bringCloserTo(origin, target):
    diff = locationdiff(origin,target)
    return origin[0]+sign(diff[0]),origin[1]+sign(diff[1])
#returns the list of coordinates between two coordinates
def getPath(origin, target):
    origin = bringCloserTo(origin,target)
    while origin!=target:
        yield origin
        origin = bringCloserTo(origin,target)
#represents a single piece
class piece:
    def __init__(self,token, encodingslot, encodingFlag = 0):
        self.token = token
        self.location = None
        self.firstMoveTime = None
        self.encodingslot = encodingslot
        self.encodingFlag = encodingFlag
    #a letter representing the piece
    @staticmethod
    def sign():
        return '?'
    #whether the piece can move to specified location, not to be overidden
    def validMove(self, targetLocation, board):
        if not board.occupant(targetLocation) is None:
            return False
        return self.canMove(targetLocation, board)
    #whether the piece can move to specified location, to be overidden
    def canMove(self,targetLocation,board):
        raise NotImplementedError
    #whether the piece can eat at specified location, not to be overidden
    def validEat(self,targetLocation,targetPiece,board):
        if targetPiece.token == self.token:
            return False
        return self.canEat(targetLocation,targetPiece,board)
    #whether the piece can eat at specified location, to be overidden
    def canEat(self,targetLocation,targetPiece,board):
        if targetPiece.location != targetLocation:
            return False
        return self.canMove(targetLocation,board)
    #treat mechanics as though time has passed
    def advanceTime(self):
        if not self.firstMoveTime is None:
            self.firstMoveTime+=1
    #move to new location
    def move(self,newLocation):
        self.location = newLocation
        if self.firstMoveTime is None:
            self.firstMoveTime = 0
    def __repr__(self):
        return "{}s {} at {}".format(self.token,self.sign(),self.location)
class pawn(piece):
    def __init__(self,token,direction, encodingslot):
        piece.__init__(self,token, encodingslot)
        self.jumpTime = None
        self.direction = direction
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location, targetLocation)
        if diff[1]!=0:
            return False
        if diff[0] == self.direction:
            return True
        if diff[0] == 2*self.direction:
            return self.firstMoveTime is None
        return False
    def canEat(self,targetLocation,targetPiece,board):
        diff = locationdiff(self.location,targetLocation)
        if diff[0] != self.direction:
            return False
        if abs(diff[1]) != 1:
            return False
        if isinstance(targetPiece,pawn) and targetPiece.jumpTime == 1:
            #chance for en passant
            eatdiff = locationdiff(self.location,targetPiece.location)
            terdiff = locationdiff(targetPiece.location, targetLocation)
            return (abs(eatdiff[1]) == 1 and eatdiff[0]==0) and (terdiff[1] == 0 and terdiff[0] == self.direction)
        return True
    def advanceTime(self):
        piece.advanceTime(self)
        if not self.jumpTime is None:
            self.jumpTime+=1
    def move(self,newLocation):
        piece.move(self,newLocation)
        diff = locationdiff(self.location,newLocation)
        if diff[1] == 2:
            self.jumpTime = 0
    @staticmethod
    def sign():
        return "P"
class knight(piece):
    def __init__(self,token, encodingslot):
        piece.__init__(self,token, encodingslot, 1 << 6)
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location,targetLocation)
        if (abs(diff[0]) + abs(diff[1])) !=3:
            return False
        return True
    @staticmethod
    def sign():
        return "H"
class king(piece):
    def __init__(self,token, encodingslot):
        piece.__init__(self,token, encodingslot)
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location,targetLocation)
        if abs(diff[0]) > 1 or abs(diff[1])>1 :
            return False
        return True
    @staticmethod
    def sign():
        return "K"
class rook(piece):
    def __init__(self,token, encodingslot):
        piece.__init__(self,token, encodingslot)
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location,targetLocation)
        if diff[0] != 0 and diff[1] != 0:
            return False
        return allClear(board,getPath(self.location,targetLocation))
    @staticmethod
    def sign():
        return "R"
class bishop(piece):
    def __init__(self,token, encodingslot):
        piece.__init__(self,token, encodingslot)
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location,targetLocation)
        if abs(diff[0]) != abs(diff[1]):
            return False
        return allClear(board,getPath(self.location,targetLocation))
    @staticmethod
    def sign():
        return "B"
class queen(piece):
    def __init__(self,token, encodingslot):
        piece.__init__(self,token, encodingslot, 1<<7)
    def canMove(self,targetLocation,board):
        diff = locationdiff(self.location,targetLocation)
        if abs(diff[0]) != abs(diff[1]) and diff[0] != 0 and diff[1] != 0:
            return False
        return allClear(board,getPath(self.location,targetLocation))
    @staticmethod
    def sign():
        return "Q"