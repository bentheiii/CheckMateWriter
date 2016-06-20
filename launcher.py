from topApp import topApp
from logicCore import LogicCore
from viewercore import viewerCore
camfps = 24
statefps = 0.5
app = topApp(None, LogicCore(lambda x:'recording/game{}.ckm'.format(x)), viewerCore(), camfps,statefps)
app.mainloop()