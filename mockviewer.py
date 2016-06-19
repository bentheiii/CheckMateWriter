import cv2

class mockViewer:
    def __init__(self):
        self.source=None
    def getFrame(self):
        return self.source.read()[1]
    def getBoard(self):
        return None
    def switchSource(self,ind):
        try:
            self.source = cv2.VideoCapture(ind)
            im = self.source.read()[0]
            if im == 0:
                raise Exception()
        except:
            return False
        return True
    def calibrate(self,empty,set):
        cv2.imwrite('empty.jpg',empty)
        cv2.imwrite('set.jpg',set)
        return False, 'not implemented'
    def close(self):
        self.source.release()