import cv2
import numpy as np

import corners_identification as ci
import points_and_colors_manipulations as pcm


class viewerCore:
    def __init__(self):
        self.source = None
        self.gray = None
        self.scale = None
        self.first = None
        self.corners = None

    # getFrame()->Image
    # get undistorted image directly from camera
    def getFrame(self):
        if self.source == None:
            print 'need to switch source first!'
            return
        try:
            ret, frame = self.source.read()
            if ret == False:
                raise Exception()
        except:
            print 'frame is NOT read correctly from capture'
            return

        return frame

    # getBoard()->8x8 board of 0 (white), 1 (black),None (vacant)
    # get board from parser, should return None in case of bad frame
    def getBoard(self):
        game_img = self.getFrame()
        if game_img == None:
            print 'error occurred when reading frame'
            return

        print "inner corners", ci.getInnerCorners(game_img)  #

        edges_img = cv2.Canny(game_img, 100, 200)
        result_matrix = np.empty([8, 8])
        k = 0
        # save square color, 0 for white, 1 for brown
        square_color = -1
        if self.first == 1:
            square_color = 1
            print "first is brown"
        else:
            square_color = 0
            print "first is white"
        none_corners = []  # debug
        corners = self.corners
        # loop over Chess board squares and mark in matrix the squares' statuses (occupied or not)
        for i in range(8):
            for j in range(8):
                # print k, k+1, k+9, k+10
                tl = corners[k][0]
                tr = corners[k + 1][0]
                bl = corners[k + 9][0]
                br = corners[k + 10][0]
                # print tl, tr, bl, br
                if pcm.isSquareEmpty(tl, br, tr, bl, edges_img):
                    result_matrix[i][j] = 0
                    square_color = 1 - square_color
                    k += 1
                    continue
                dom_color = pcm.getDominantColor(tl, br, tr, bl, game_img)
                result = pcm.getPositionStatusByDominantColor(dom_color, square_color)
                if result != 1 and result != 2:
                    result = pcm.getPositionStatusByHalfDom(tl, br, tr, bl, game_img, square_color)
                    none_corners.append(corners[k])
                    print "new result", result
                result_matrix[i][j] = result
                square_color = 1 - square_color
                k += 1
                print "########################################\n"
            square_color = 1 - square_color
            k += 1
        print result_matrix
        ci.drawCorners(game_img, none_corners)
        ci.show(game_img)
        return None

    # switchSource(int)->bool
    # switches the source to camera by number. Returns whether the connection was succesfull
    def switchSource(self, ind):
        try:
            self.source = cv2.VideoCapture(ind)
            ret = self.source.read()[0]
            if not ret:
                raise Exception()
        except:
            print 'frame is NOT read correctly from capture'
            return False
        return True

    # calibrate(Image emptyBoard, Image pieceBoard)->(bool,str)
    # recalibrate camera based on new empty board and color board, return
    # true if succeeded, if fail, return error message
    def calibrate(self, empty, initial):

        board_img = cv2.imread(empty)
        initial_img = cv2.imread(initial)
        gray = self.gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        undis_b = ci.undistort(gray, board_img)
        if undis_b is None:
            return False, 'cannot match corners in clean board image!'

        scale = self.scale = 0.7
        board_img = cv2.resize(undis_b, (0, 0), fx=scale, fy=scale)

        undis_i = ci.undistort(gray, initial_img)
        if undis_i is None:
            return False, 'cannot match corners in initial board image!'

        initial_img = cv2.resize(undis_i, (0, 0), fx=scale, fy=scale)

        innercorners = ci.getInnerCorners(board_img)
        corners = self.corners = ci.extrapolateOuterCorners(innercorners)

        ci.drawCorners(board_img, corners)  #
        ci.show(board_img)  #

        # set 2 opposite colors of board squares
        pcm.BROWN_RGB, pcm.WHITE_RGB, self.first = pcm.setRGBRanges(corners, board_img)
        print "BROWN_RGB", pcm.BROWN_RGB
        print "WHITE_RGB", pcm.WHITE_RGB

        # set 2 opposite colors of board pieces
        pcm.FIRST_PIECE_RGB, pcm.SECOND_PIECE_RGB = pcm.initialPiecesRGB(corners, initial_img)
        print "FIRST_PIECE_RGB", pcm.FIRST_PIECE_RGB
        print "SECOND_PIECE_RGB", pcm.SECOND_PIECE_RGB

        return True, 'Done!'

    # close()->None
    # closes the object
    def close(self):
        self.source.release()
        cv2.destroyAllWindows()