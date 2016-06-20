import numpy as np
import cv2
import argparse

import corners_identification as ci
import points_and_colors_manipulations as pcm
#import rgb

parser = argparse.ArgumentParser(description = "identify pieces on Checkmate board")
parser.add_argument("-cb", "--clean_board", action="store", default="cb.jpg", help="clean Checkmate board")
parser.add_argument("-ib", "--initial_board", action="store", default="ib.jpg", help="initial Checkmate board")
parser.add_argument("-gb", "--game_board", action="store", default="gb.jpg", help="one frame of Checkmate game board")
args = parser.parse_args()

board_img = cv2.imread(args.clean_board)
initial_img = cv2.imread(args.initial_board)

gray = cv2.cvtColor(board_img,cv2.COLOR_BGR2GRAY)
#ci.show(gray)
undis_b = ci.undistort(gray,board_img)
if undis_b is None:
    print "cannot match corners in clean board image!"
    exit(1)

scale = 1
board_img = cv2.resize(undis_b,(0,0),fx = scale, fy = scale)

undis_i = ci.undistort(gray,initial_img)
if undis_i is None:
    print "cannot match corners in initial board image!"
    exit(1)
initial_img = cv2.resize(undis_i,(0,0),fx = scale, fy = scale)

#ci.show(board_img)

innercorners = ci.getInnerCorners(board_img)
corners = ci.extrapolateOuterCorners(innercorners)

ci.drawCorners(board_img, corners) #
ci.show(board_img) #

# set 2 opposite colors of board squares
pcm.BROWN_RGB, pcm.WHITE_RGB, first = pcm.setRGBRanges(corners, board_img)
print "BROWN_RGB",pcm.BROWN_RGB
print "WHITE_RGB", pcm.WHITE_RGB

# set 2 opposite colors of board pieces
pcm.FIRST_PIECE_RGB, pcm.SECOND_PIECE_RGB = pcm.initialPiecesRGB(corners, initial_img)
print "FIRST_PIECE_RGB", pcm.FIRST_PIECE_RGB
print "SECOND_PIECE_RGB", pcm.SECOND_PIECE_RGB


#var = raw_input("Please enter something: ") ######

game_img = cv2.imread(args.game_board)
undis_g = ci.undistort(gray,game_img)
if undis_g is None:
    print "cannot match corners in game board image!"
    exit(1)
game_img = cv2.resize(undis_g,(0,0),fx = scale, fy = scale)
edges_img = cv2.Canny(game_img,70,120) #

###
#rgb.test(corners, initial_img)
# var = raw_input("Please enter something: ")

# var = raw_input("Please enter something: ")
print "inner corners", ci.getInnerCorners(game_img) #
result_matrix = np.empty([8, 8])
k = 0
# save square color, 0 for white, 1 for brown
square_color = -1
if first == 1:
    square_color = 1
    print "first is brown"
else:
    square_color = 0
    print "first is white"
none_corners = [] #debug
# loop over Chess board squares and mark in matrix the squares' statuses (occupied or not)
for i in range(8):
    for j in range(8):
        # print k, k+1, k+9, k+10
        tl = corners[k][0]
        tr = corners[k+1][0]
        bl = corners[k+9][0]
        br = corners[k+10][0]
        # print tl, tr, bl, br
        if pcm.isSquareEmpty(tl, br, tr, bl, edges_img):
             result_matrix[i][j] = 0
             square_color = 1-square_color
             k += 1
             continue
        dom_color = pcm.getDominantColor(tl, br, tr, bl, game_img)
        result = pcm.getPositionStatusByDominantColor(dom_color, square_color)
        if result!=1 and result!=2 :
            result = pcm.getPositionStatusByHalfDom(tl, br, tr, bl, game_img, square_color)
            none_corners.append(corners[k])
            print "new result", result
        result_matrix[i][j] = result
        square_color = 1-square_color
        k += 1
        print "########################################\n"
    square_color = 1-square_color
    k += 1
print result_matrix
ci.drawCorners(game_img, none_corners)
ci.show(game_img)