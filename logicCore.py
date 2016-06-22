from moveinterpreter import moveInterpreter, moveTypes
from validator import validator
from gameRecorder import openRecorder
import os.path

#takes a boardstate and returns its string representation, capital is black, lowercase is white, ':' is vacant
def strSigns(state, wToken, size=8, transform=False):
    ret = []
    for x in xrange(size):
        for y in xrange(size):
            if state.occupant(x,y,transform) is None:
                ret.append(":")
            else:
                sign = str(state.occupant(x,y,transform).sign())
                if wToken == state.occupant(x,y,transform).token:
                    sign = str.lower(sign)
                ret.append(sign)
        ret.append("\n")
    return "".join(ret)

class LogicCore:
    def __init__(self,filenameGenerator):
        self.recorder = None
        self.validator = None
        self.interpreter = None
        self.fileGenerator = filenameGenerator
        self.gameName = None
    #a string for the next player that can play
    def nextPlayer(self):
        ret =  self.validator.nextPlay()
        if ret is None:
            return 'Anyone'
        if ret == self.validator.board.tokenW:
            return 'White'
        return 'Black'
    #returns a string representing the current valid game state
    def getState(self):
        return strSigns(self.validator.board,self.validator.board.tokenW,transform=True)
    #tries to change the current board to new state, returns None if legal, otherwise returns error message
    def mutate(self,board,promotionvalue):
        move = self.interpreter.nextmove(board)
        valid, valValue = self.validator.isValid(move)
        if valid:
            self.interpreter.commit(board)
            self.validator.Commit(move,valValue,promotionvalue)
            self.recorder.record(self.validator.board,move,self.validator.nextPlay())
            return None
        elif move.type == moveTypes.no:
            return "No Move Detected"
        else:
            return "Illegal move!"
    #initializes components towards new game with new name
    def startNewGame(self):
        i = 0
        while os.path.isfile(self.fileGenerator(i)):
            i+=1
        self.gameName = self.fileGenerator(i)
        self.recorder = openRecorder(self.gameName,0)
        self.interpreter = moveInterpreter()
        self.validator = validator(0,1)
    #get current game name
    def getGameName(self):
        return self.gameName
    #initiate rollback (if possible)
    def rollBack(self):
        if len(self.validator.boards) == 0:
            return
        self.validator.rollBack()
        self.interpreter.rollBack()
        self.recorder.rollBack()