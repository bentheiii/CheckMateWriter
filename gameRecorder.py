from encoding import encodeState, ENCODINGSIZE
import os
import platform

class osStream:
    def __init__(self,filepath):
        if platform.system()=='Windows':
            raise Exception("windows is not supported")
        else:
            self.file = open(filepath,'wb',0)
    def write(self,text):
        self.file.write(text)
    def close(self):
        self.file.close()
    def seek(self,offset,whence):
        self.file.seek(offset,whence)
    def truncate(self):
        self.file.truncate()

def openRecorder(filepath,whiteToken):
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return gameRecorder(osStream(filepath),whiteToken)

class gameRecorder:
    def __init__(self,stream,whiteToken):
        self._file = stream
        self.tokenW = whiteToken
    def record(self,board,move,nextplay):
        towrite = bytearray(encodeState(board,move,self.tokenW,nextplay))
        self.write(towrite)
    def write(self,data):
        self._file.write(data)
    def __del__(self):
        self._file.close()
    def rollBack(self,size = ENCODINGSIZE):
        self._file.seek(-size, os.SEEK_END)
        self._file.truncate()