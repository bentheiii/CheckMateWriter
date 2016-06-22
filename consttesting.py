#testing program for live camera

import cv2
import argparse
import sys
import corners_identification as ci
import viewercore

parser = argparse.ArgumentParser(description = "identify pieces on Checkmate board")
parser.add_argument("-cb", "--clean_board", action="store", default="cb.jpg", help="clean Checkmate board")
parser.add_argument("-ib", "--initial_board", action="store", default="ib.jpg", help="initial Checkmate board")
parser.add_argument("-gb", "--game_board", action="store", default="gb.jpg", help="one frame of Checkmate game board")
class expando:
    pass
args = expando()
args.clean_board = sys.argv[1]+r"/empty.jpg"
args.initial_board = sys.argv[1]+r"/set.jpg"

board_img = cv2.imread(args.clean_board)
initial_img = cv2.imread(args.initial_board)

core = viewercore.viewerCore()
_,dat = core.cabEmpty(board_img)
core.cabSet(dat,initial_img)

while True:
    source = cv2.VideoCapture(1)
    board_img = source.read()[1]
    source.release()
    board = core.getBoard(board_img)
    print board
    ci.show(board_img)