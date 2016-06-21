import numpy as np
import cv2
import argparse
import sys
import corners_identification as ci
import points_and_colors_manipulations as pcm
import viewercore
#import rgb

parser = argparse.ArgumentParser(description = "identify pieces on Checkmate board")
parser.add_argument("-cb", "--clean_board", action="store", default="cb.jpg", help="clean Checkmate board")
parser.add_argument("-ib", "--initial_board", action="store", default="ib.jpg", help="initial Checkmate board")
parser.add_argument("-gb", "--game_board", action="store", default="gb.jpg", help="one frame of Checkmate game board")
#args = parser.parse_args()
class expando:
    pass
args = expando()
args.clean_board = sys.argv[1]+r"/empty.jpg"
args.initial_board = sys.argv[1]+r"/set.jpg"
zind = sys.argv[2] if len(sys.argv) > 2 else raw_input("enter z index:\n")
args.game_board = sys.argv[1]+r"/z"+zind+".jpg"

board_img = cv2.imread(args.clean_board)
initial_img = cv2.imread(args.initial_board)
game_img = cv2.imread(args.game_board)

core = viewercore.viewerCore()
_,dat = core.cabEmpty(board_img)
core.cabSet(dat,initial_img)
board = core.getBoard(game_img)
print board
ci.show(game_img)
