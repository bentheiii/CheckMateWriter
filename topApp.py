import Tkinter
import Image
import cv2
import ImageTk
from moveinterpreter import board
from captureDiag import captureDiag

#todo: hookup, rollback

class topApp(Tkinter.Tk):
    def __init__(self, parent, logic, viewer, camfps = 30, statefps = 4):
        Tkinter.Tk.__init__(self,parent)

        #stages:
        #0- no cam
        #1- no calibration
        #2- no game
        #3- game ready
        self.stage = 0

        self.camfps = camfps
        self.statefps = statefps
        self.parent = parent
        self.logic = logic
        self.viewer = viewer
        self.camind = -1
        self.title("CheckMate Writer")

        self.initialize()

    def initialize(self):
        self.grid()

        self.camLabel = Tkinter.Label(self)
        self.camLabel.grid(column=0, row=1, columnspan=4, rowspan=3, sticky='ENSW')

        self.promotionval = Tkinter.StringVar()

        radioQ = Tkinter.Radiobutton(self,text='promote to queen', variable = self.promotionval, value = 'Q')
        radioQ.grid(column=0,row=0,columnspan=2)
        radioQ.select()
        radioH = Tkinter.Radiobutton(self,text='promote to knight', variable = self.promotionval, value = 'H')
        radioH.grid(column=2,row=0,columnspan=2)

        self.boardState = Tkinter.StringVar()
        boardStateLabel= Tkinter.Label(self, textvariable = self.boardState)
        boardStateLabel.grid(column=4,row=0,columnspan=2,rowspan=2)

        self.boardState.set('++++++++\n++++++++\n++++++++\n++++++++\n++++++++\n++++++++\n++++++++\n++++++++\n')
        buttonheight = 4
        rollButton = Tkinter.Button(self, text='RollBack', height = buttonheight, command=self.rollback)
        rollButton.grid(column=4,row=2,columnspan=2)

        calibrateButton = Tkinter.Button(self,text='Calibrate', height = buttonheight,command=self.calibrate)
        calibrateButton.grid(column=4,row=3,columnspan=2)

        newGameButton = Tkinter.Button(self,text='New Game', height = buttonheight,command = self.newGame)
        newGameButton.grid(column=6,row=0,columnspan=3)

        switchButton = Tkinter.Button(self,text='Switch Camera', height = buttonheight,command = self.switchCam)
        switchButton.grid(column=6,row=1,columnspan=3)

        self.message = Tkinter.StringVar()

        messageLabel = Tkinter.Label(self,textvariable=self.message)
        messageLabel.grid(column=6,row=2,rowspan=2,columnspan=3)

        self.seekcam()

        self.loopCam()
        self.loopState()
    def rollback(self):
        if self.stage >= 3:
            self.logic.rollBack()
    def frameDiag(self,prompt,title):
        d = captureDiag(self,self.viewer,prompt,title)
        self.wait_window(d)
        return d.value
    def calibrate(self):
        if self.stage < 1:
            return
        empty = self.frameDiag('Click the button when the board is empty and in view','Empty Board')
        if empty is None:
            return
        result, message = self.viewer.cabEmpty(empty)
        if not result:
            self.setMessage('Calibration Failed:\n{}'.format(message))
        setboard = self.frameDiag('Click the button when the board is in its starting position', 'starting board')
        if setboard is None:
            return
        result,message = self.viewer.cabSet(empty,message,setboard)
        if result:
            self.setMessage('Calibration sucessfull')
            self.stage = max(self.stage,2)
        else:
            self.setMessage('Calibration failed:\n{}'.format(message))
    def setMessage(self,text,displayGameName=False):
        if displayGameName:
            text = self.logic.getGameName()+'\n\n\n'+text
        self.message.set(text)
    def seekcam(self):
        while True:
            self.camind+=1
            sucess = self.viewer.switchSource(self.camind)
            if sucess:
                self.stage = max(self.stage,1)
                return
            elif self.camind == 0:
                raise Exception('could not connect to cam 0')
            else:
                self.camind = -1
    def newGame(self):
        if self.stage < 2:
            return
        self.logic.startNewGame()
        self.setMessage('New game started!',True)
        self.stage = 3
    def switchCam(self):
        self.seekcam()
    def loopState(self):
        self.stateUpdate()
        self.after(int(1000/ self.statefps), self.loopState)
    def stateUpdate(self):
        if self.stage >=3:
            npboard = self.viewer.getBoard()
            if npboard is None:
                self.setMessage("bad frame")
                return
            mut = self.logic.mutate(board.fromnp(npboard),self.promotionval)
            if mut is not None:
                self.setMessage(mut, True)
            else:
                self.boardState.set(self.logic.getState())
    def loopCam(self):
        self.camUpdate()
        self.after(1000 / self.camfps, self.loopCam)
    def camUpdate(self):
        frame = self.viewer.getFrame()
        frame = cv2.resize(frame,(400,300))
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.imgtk = imgtk
        self.camLabel.configure(image=imgtk)