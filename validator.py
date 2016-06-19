from pieces import *
from boardstate import boardState, coortransform, allClear
from moveinterpreter import moveTypes
import copy

def containerequals(item1,item2):
    for i in item1:
        if not i in item2:
            return False
    return True

class validator:
    def __init__(self,whitetoken,blacktoken,size=8):
        self.boards = [boardState(whitetoken,blacktoken,size)]
        #dictates which token should not play next, None means anyone can play next (in case of assignment)
        self.prevTurn = None

    @property
    def board(self):
        return self.boards[-1]
    @board.setter
    def board(self,value):
        self.boards.append(copy.deepcopy(value))

    def ValidateAppear(self,move):
        whites = filter(lambda x:x.kind==self.board.tokenW,move.appears)
        #whites = []
        #for c in move.appears:
        #    if c.kind == self.board.tokenW:
        #        whites.append(c)
        blacks = filter(lambda x:x.kind==self.board.tokenB,move.appears)
        if (not self.board.pieces is None) or len(move.appears) != 32 \
        or len(whites) != 16 \
        or len(blacks) != 16:
            return False, None
        #identify corners
        #(0,0)
        NWCorner = None
        #(0,size-1)
        NECorner = None
        #(size-1,0)
        SWCorner = None
        #(size-1,size-1)
        SECorner = None

        for appear in move.appears:
            if appear.coordinates==(0,0):
                NWCorner = appear.kind
            elif appear.coordinates==(0,self.board.size-1):
                NECorner = appear.kind
            elif appear.coordinates==(self.board.size-1,0):
                SWCorner = appear.kind
            elif appear.coordinates==(self.board.size-1,self.board.size-1):
                SECorner = appear.kind

        #all corners have to be filled
        if NWCorner is None or NECorner is None or SWCorner is None or SECorner is None:
            return False, None

        if NWCorner == NECorner and SWCorner == SECorner and NWCorner != SECorner:
            #white on north (0,x), (1,x), black on south (size-1,x) (size-2,x) (or reversed)
            for appear in move.appears:
                if appear.coordinates[0] <= 1:
                    if appear.kind == NWCorner:
                        continue
                    else:
                        return False, None
                if appear.coordinates[0] >= self.board.size-2:
                    if appear.kind == SECorner:
                        continue
                    else:
                        return False, None
                return False, None
        elif NWCorner == SWCorner and NECorner == SECorner and NWCorner != SECorner:
            #white on west (x,0), (x,1), black on south (x,size-1) (x,size-2) (or reversed)
            for appear in move.appears:
                if appear.coordinates[1] <= 1:
                    if appear.kind == NWCorner:
                        continue
                    else:
                        return False, None
                if appear.coordinates[1] >= self.board.size-2:
                    if appear.kind == SECorner:
                        continue
                    else:
                        return False, None
                return False, None
        else:
            return False,None
        #get the transform
        if NWCorner==NECorner:
            if NWCorner==self.board.tokenW:
                return True,0
            else:
                return True,2
        if NWCorner==SWCorner:
            if NWCorner==self.board.tokenW:
                return True,3
            else:
                return True,1
    def ValidateCastling(self,move):
        if len(move.appears) != 2:
            return False,None
        piece1 = self.board.occupant(move.disappears[0].coordinates,transformcoor=True)
        piece2 = self.board.occupant(move.disappears[1].coordinates,transformcoor=True)
        if isinstance(piece1,king) and isinstance(piece2,rook):
            k = piece1
            r = piece2
        elif isinstance(piece2,king) and isinstance(piece1,rook):
            k = piece2
            r = piece1
        else:
            return False, None
        if k.firstMoveTime is not None or r.firstMoveTime is not None:
            return False, None
        if not allClear(self.board,getPath(k.location,r.location)):
            return False, None
        if r.location[1] == 7:
            #big castling
            newlocs = ((r.location[0],6),(r.location[0],5))
        else:
            #small castling
            newlocs = ((r.location[0],2),(r.location[0],3))
        if not containerequals(map(lambda x:coortransform(x.coordinates,self.board.transform,self.board.size),move.appears),newlocs):
            return False, None
        return True,(k,r,newlocs[0],newlocs[1])
    def commitAppear(self,transform):
        self.board.transform = transform

        self.board.insertPiece(rook(self.board.tokenW,8),(0,0))
        self.board.insertPiece(rook(self.board.tokenW,9),(0,7))
        self.board.insertPiece(knight(self.board.tokenW,10),(0,1))
        self.board.insertPiece(knight(self.board.tokenW,11),(0,6))
        self.board.insertPiece(bishop(self.board.tokenW,12),(0,2))
        self.board.insertPiece(bishop(self.board.tokenW,13),(0,5))
        self.board.insertPiece(queen(self.board.tokenW,14),(0,3))
        self.board.insertPiece(king(self.board.tokenW,15),(0,4))

        self.board.insertPiece(rook(self.board.tokenB,24),(7,0))
        self.board.insertPiece(rook(self.board.tokenB,25),(7,7))
        self.board.insertPiece(knight(self.board.tokenB,26),(7,1))
        self.board.insertPiece(knight(self.board.tokenB,27),(7,6))
        self.board.insertPiece(bishop(self.board.tokenB,28),(7,2))
        self.board.insertPiece(bishop(self.board.tokenB,29),(7,5))
        self.board.insertPiece(queen(self.board.tokenB,30),(7,3))
        self.board.insertPiece(king(self.board.tokenB,31),(7,4))

        for i in xrange(8):
            self.board.insertPiece(pawn(self.board.tokenW,1,i),(1,i))
            self.board.insertPiece(pawn(self.board.tokenB,-1,i+16),(6,i))

        self.advanceTurn(None)
    def commitCastling(self,valValue):
        k,r,kloc,rloc = valValue
        self.board.assign(None,k.location)
        k.move(kloc)
        self.board.assign(k,k.location)
        self.board.assign(None,r.location)
        r.move(rloc)
        self.board.assign(r,r.location)
        self.advanceTurn(k.token)
    def hasTurn(self,playToken):
        if self.prevTurn is None:
            return True
        return self.prevTurn != playToken
    def nextPlay(self):
        if self.prevTurn is None:
            return None
        if self.board.tokenB == self.prevTurn:
            return self.board.tokenW
        else:
            return self.board.tokenB
    def isValid(self,move):
        if move.type == moveTypes.appear:
            return self.ValidateAppear(move)
        if self.board.pieces is None:
            return False,None
        if move.type == moveTypes.regular:
            mover = self.board.occupant(move.disappears[0].coordinates, transformcoor=True)
            if not self.hasTurn(mover.token):
                return False, None
            board = self.board if mover.token==self.board.tokenW else self.board.reverse()
            targetcoor = coortransform(move.appears[0].coordinates,self.board.transform,self.board.size)
            if not mover.validMove(targetcoor,board):
                return False, None
            promote = False
            if isinstance(mover, pawn) and targetcoor[0] == (7 if mover.direction == 1 else 0):
                promote = True
            return True, (mover, promote)
        if move.type == moveTypes.eat or move.type == moveTypes.irregularEat:
            eaten = self.board.occupant(move.disappears[0].coordinates if move.disappears[1].kind == move.appears[0].kind else move.disappears[1].coordinates, transformcoor=True)
            eatertarloc = coortransform(move.appears[0].coordinates,self.board.transform,self.board.size)
            eater = self.board.occupant(move.disappears[0].coordinates if move.disappears[0].kind == move.appears[0].kind else move.disappears[1].coordinates, transformcoor=True)
            if not self.hasTurn(eater.token):
                return False, None
            board = self.board if eater.token==self.board.tokenW else self.board.reverse()
            if not eater.validEat(eatertarloc,eaten,board):
                return False,None
            promote = False
            if isinstance(eater, pawn) and eatertarloc[0] == (7 if eater.direction == 1 else 0):
                promote = True
            return True,(eater,eaten,promote)
        if move.type == moveTypes.multiMove:
            return self.ValidateCastling(move)
        return False, None
    def advanceTurn(self,prevToken):
        self.prevTurn = prevToken
    def Commit(self,move,validationValue,promotionvalue = 'Q'):
        if move.type == moveTypes.appear:
            self.commitAppear(validationValue)
        elif move.type == moveTypes.regular:
            mover = validationValue[0]
            self.board.assign(None,mover.location)
            mover.move(coortransform(move.appears[0].coordinates,self.board.transform,self.board.size))
            self.board.assign(mover,mover.location)
            self.advanceTurn(mover.token)
            if validationValue[1] is True:
                self.promote(mover,promotionvalue)
        elif move.type == moveTypes.eat or move.type == moveTypes.irregularEat:
            eater, eaten = validationValue
            self.board.assign(None,eaten.location)
            self.board.pieces.remove(eaten)
            self.board.assign(None,eater.location)
            eater.move(coortransform(move.appears[0].coordinates,self.board.transform,self.board.size))
            self.board.assign(eater,eater.location)
            self.advanceTurn(eater.token)
            if validationValue[2] is True:
                self.promote(eater,promotionvalue)
        elif move.type == moveTypes.multiMove:
            self.commitCastling(validationValue)
        self.board.advance()
    def rollBack(self):
        self.boards = self.board[:-1]
    def promote(self, propawn, promoteSign):
        proclass = queen if promoteSign=='Q' else knight
        self.board.pieces.remove(propawn)
        self.board.insertPiece(proclass(propawn.token,propawn.encodingslot),propawn.location)