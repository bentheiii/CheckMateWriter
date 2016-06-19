import Tkinter

class captureDiag(Tkinter.Toplevel):
    def __init__(self,parent,viewer,prompt,title):
        Tkinter.Toplevel.__init__(self,parent)
        self.transient(parent)

        self.prompt = prompt
        self.viewer = viewer
        self.title(title)
        self.value = None

        self.initialize()

        self.grab_set()

    def initialize(self):
        self.grid()

        label = Tkinter.Label(self,text=self.prompt)
        label.grid(row=0,column=0)

        button = Tkinter.Button(self,text='Capture',command=self.click)
        button.grid(row=1,column=0)


    def click(self):
        self.value = self.viewer.getFrame()
        self.destroy();