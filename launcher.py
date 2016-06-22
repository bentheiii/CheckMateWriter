from topApp import topApp
from logicCore import LogicCore
from viewercore import viewerCore
import sys
camfps = 24
statefps = 0.5
cabfold = None if len(sys.argv) <= 1 else sys.argv[1]
app = topApp(None, LogicCore(lambda x:'recording/game{}.ckm'.format(x)), viewerCore(), camfps,statefps, cabfold)
app.mainloop()