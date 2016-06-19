from moveinterpreter import moveInterpreter
from validator import validator
from gameRecorder import openRecorder
import os.path

def strTokens(state,size=8,transform=False):
    ret = []
    for x in xrange(size):
        for y in xrange(size):
            if state.occupant(x,y,transform) is None:
                ret.append(":")
            else:
                ret.append(state.occupant(x,y,transform).token)
        ret.append("\n")
    return "".join(ret)

class LogicCore:
    def __init__(self,filenameGenerator):
        self.recorder = None
        self.validator = None
        self.interpreter = None
        self.fileGenerator = filenameGenerator
        self.gameName = None
    def getState(self):
        return strTokens(self.validator.board)
    def mutate(self,board,promotionvalue):
        move = self.interpreter.nextmove(board)
        valid, valValue = self.validator.isValid(move)
        if valid:
            self.interpreter.commit(board)
            self.validator.Commit(move,valValue,promotionvalue)
            self.recorder.record(self.validator.board,move,self.validator.nextPlay())
            return None
        else:
            return "Illegal move!"
    def startNewGame(self):
        i = 0
        while os.path.isfile(self.fileGenerator(i)):
            i+=1
        self.gameName = self.fileGenerator(i)
        self.recorder = openRecorder(self.gameName,0)
        self.interpreter = moveInterpreter()
        self.validator = validator(0,1)
    def getGameName(self):
        return self.gameName
    def rollBack(self):
        if len(self.validator.boards) == 0:
            return
        self.validator.rollBack()
        self.interpreter.rollBack()
        self.recorder.rollBack()