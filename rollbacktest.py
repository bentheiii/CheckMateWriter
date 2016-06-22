#tests recorder's ability for rollback

import gameRecorder

rec = gameRecorder.openRecorder("testing/recordtest.txt",1)

rec.write("0123456789abcdefghijklmnopqrstuvwxyz")
rec.rollBack(3)
del rec