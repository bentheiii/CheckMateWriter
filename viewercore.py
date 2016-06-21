import cv2

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
        if self.source is None:
            # print 'need to switch source first!'
            return
        try:
            ret, frame = self.source.read()
            if ret == False:
                raise Exception()
        except:
            # print 'frame is NOT read correctly from capture'
            return
        return frame

    # getBoard()->8x8 board of 0 (white), 1 (black),None (vacant)
    # get board from parser, should return None in case of bad frame
    def getBoard(self,frame):
        if self.gray is None:
            # print 'need to calibrate camera first!'
            return
        undis_g = ci.undistort(self.gray, frame)
        if undis_g is None:
            # print "cannot match corners in game board image!"
            return
        scale = self.scale
        game_img = cv2.resize(undis_g, (0, 0), fx=scale, fy=scale)
        if game_img == None:
            # print 'error occurred when reading frame'
            return

        # print "inner corners", ci.getInnerCorners(game_img)  #

        edges_img = cv2.Canny(game_img, 70, 120)
        #ci.show(edges_img)
        # save square color, 0 for white, 1 for brown
        square_color = -1
        if self.first == 1:
            square_color = 1
        else:
            square_color = 0
        return pcm.getResultsForOneFrame(game_img, edges_img, self.corners, square_color)


    # switchSource(int)->bool
    # switches the source to camera by number. Returns whether the connection was succesfull
    def switchSource(self, ind):
        try:
            self.source = cv2.VideoCapture(ind)
            ret = self.source.read()[0]
            if ret == False:
                raise Exception()
        except:
            # print 'frame is NOT read correctly from capture'
            return False
        return True


    def cabEmpty(self,empty):
        board_img = empty
        gray = self.gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        undis_b = ci.undistort(gray, board_img)
        if undis_b is None:
            return False, 'cannot match corners in clean board image!'

        self.scale = 0.7
        board_img = cv2.resize(undis_b, (0, 0), fx=self.scale, fy=self.scale)


        innercorners = ci.getInnerCorners(board_img)
        if innercorners is None:
            return False, "cannot identify corners"
        corners = self.corners = ci.extrapolateOuterCorners(innercorners)
        return True, (corners, board_img, gray)
    # calibrate(Image emptyBoard, Image pieceBoard)->(bool,str)
    # recalibrate camera based on new empty board and color board, return
    # true if succeeded, if fail, return error message
    def cabSet(self, dat, initial):
        corners, board_img, gray = dat
        initial_img = initial

        undis_i = ci.undistort(gray, initial_img)
        if undis_i is None:
            return False, 'cannot match corners in initial board image!'

        initial_img = cv2.resize(undis_i, (0, 0), fx=self.scale, fy=self.scale)

        #ci.drawCorners(board_img, corners)  #
        #ci.show(board_img)  #

        # set 2 opposite colors of board squares
        #pcm.BROWN_RGB, pcm.WHITE_RGB, self.first = pcm.setRGBRanges(corners, board_img)
        pcm.CELL_RANGE_MATRIX = pcm.SetCellRanges(corners,board_img)
        # print "BROWN_RGB", pcm.BROWN_RGB
        # print "WHITE_RGB", pcm.WHITE_RGB

        # set 2 opposite colors of board pieces
        pcm.FIRST_PIECE_RGB, pcm.SECOND_PIECE_RGB = pcm.initialPiecesRGB(corners, initial_img)
        # print "FIRST_PIECE_RGB", pcm.FIRST_PIECE_RGB
        # print "SECOND_PIECE_RGB", pcm.SECOND_PIECE_RGB

        return True, 'Done!'

    # close()->None
    # closes the object
    def close(self):
        self.source.release()
        cv2.destroyAllWindows()