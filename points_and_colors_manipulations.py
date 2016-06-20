import cv2
import numpy as np
from sklearn.cluster import KMeans

import corners_identification as ci

# define RGB colors ranges
BROWN_RGB = [(50, 20, 0), (110, 70, 40)]
WHITE_RGB = [(150, 140, 120), (200, 190, 185)]

FIRST_PIECE_RGB = [(0,0,0), (60,60,60)]
SECOND_PIECE_RGB = [(60,60,60), (180,180,180)]


#
def getMinPixelFromList(pixels_list):
    R = []
    G = []
    B = []
    for pixel in pixels_list:
        R.append(pixel[0])
        G.append(pixel[1])
        B.append(pixel[2])
    return (min(R), min(G), min(B))

#
def getMaxPixelFromList(pixels_list):
    R = []
    G = []
    B = []
    for pixel in pixels_list:
        R.append(pixel[0])
        G.append(pixel[1])
        B.append(pixel[2])
    return (max(R), max(G), max(B))

#
def getAveragePixel(pixels_list):
    r, g, b = 0, 0, 0
    count = 0
    for pixel in pixels_list:
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]
        count += 1
    return ((r/count), (g/count), (b/count))

# p1<p2 -> 1, p1>p2 -> 2, othewise -> 0
def comparePixels(p1, p2):
    if p1[0] < p2[0] and p1[1] < p2[1] and p1[2] < p2[2]:
        return 1
    if p1[0] > p2[0] and p1[1] > p2[1] and p1[2] > p2[2]:
        return 2
    return 0

# 
def getFilteredList(pixels_list, epsilon):
    avg = getAveragePixel(pixels_list)
    # print pixels_list, "avg", avg
    tmp = 1/epsilon
    avg_min = (tmp*avg[0], tmp*avg[1], tmp*avg[2])
    avg_max = (epsilon*avg[0], epsilon*avg[1], epsilon*avg[2])
    pixels_list = [p for p in pixels_list if comparePixels(p, avg_min) != 1 and comparePixels(p, avg_max) != 2]
    # print "after filtered", pixels_list
    return pixels_list

# get min and max pixel from pixels list with noisy pixels excluded
def getMinMaxFilteredPixel(pixels_list):
    pixels_list = getFilteredList(pixels_list, 3)
    min_pixel = getMinPixelFromList(pixels_list)
    max_pixel = getMaxPixelFromList(pixels_list)
    return min_pixel, max_pixel

########################################################

# 1 for horizontal board thus two top and bottom rows are placed with pieces at the begining
# 0 for vertical board thus two right and left rows are placed with pieces at the begining
def resolveBoardDirection(corners, initial_img):
    edges_img = cv2.Canny(initial_img,100,200) #
    none_corners = [] #debug
    count_empty = 0
    # sample from two middle rows
    k = 27
    for i in [3, 4]:
        # sample first and last columns
        for j in [0, 7]:
            h = k+j
            tl = corners[h][0]
            tr = corners[h+1][0]
            bl = corners[h+9][0]
            br = corners[h+10][0]
            none_corners.append(corners[h]) #
            if isSquareEmpty(tl, br, tr, bl, edges_img):
                count_empty = count_empty +1      
        k += 9

    #ci.drawCorners(edges_img, none_corners)
    #ci.show(edges_img)
    print "count empty from both side of board", count_empty
    if count_empty > 2:
        return 1
    else:
        return 0

########################################################
# return piceces' black color range and white color range
def initialPiecesRGB(corners, initial_img):
    # sample colors in squares' centers of two rows in each side of initial board, to get pieces' two colors

    centers_pixels_list1 = []
    centers_pixels_list2 = []
    # points for debug only
    centers_points = []
    k = 0
    if resolveBoardDirection(corners, initial_img) == 1:
        # board is horizontal thus two top and bottom rows are placed with pieces at the begining
    
        # sample from two first rows
        for i in [0,1]:
            for j in range(8):
                tl = corners[k][0]
                tr = corners[k+1][0]
                bl = corners[k+9][0]
                br = corners[k+10][0]
                # print tl, tr, bl, br
                centers_pixels_list1.append(getDominantColor(tl, br, tr, bl, initial_img))
                centers_points.append(corners[k]) # debug
                k += 1
            k += 1

        # increase k by 8 + 1 steps in each row till 6th row = 9*4
        k += 36

        # sample from two last rows
        for i in [6,7]:
            for j in range(8):
                tl = corners[k][0]
                tr = corners[k+1][0]
                bl = corners[k+9][0]
                br = corners[k+10][0]
                # print tl, tr, bl, br
                centers_pixels_list2.append(getDominantColor(tl, br, tr, bl, initial_img))
                centers_points.append(corners[k]) # debug
                k += 1
            k += 1
    else:
        # board is vertical thus two right and left rows are placed with pieces at the begining

        # sample from two right columns
        for i in range(8):
            for j in [0,1]:
                h = j+k
                tl = corners[h][0]
                tr = corners[h+1][0]
                bl = corners[h+9][0]
                br = corners[h+10][0]
                # print tl, tr, bl, br
                centers_pixels_list1.append(getDominantColor(tl, br, tr, bl, initial_img))
                centers_points.append(corners[h]) # debug
            k += 9

        # sample from two left columns
        k = 0
        for i in range(8):
            for j in [6, 7]:
                h = ((-1)*(6-j))+k
                tl = corners[h][0]
                tr = corners[h+1][0]
                bl = corners[h+9][0]
                br = corners[h+10][0]
                # print tl, tr, bl, br
                centers_pixels_list2.append(getDominantColor(tl, br, tr, bl, initial_img))
                centers_points.append(corners[h]) # debug
            k += 9

    # perform noise filtering of unrelevant pixels
    min1, max1 = getMinMaxFilteredPixel(centers_pixels_list1)
    min2, max2 = getMinMaxFilteredPixel(centers_pixels_list2)

    avg1 = getAveragePixel(centers_pixels_list1)#debug
    avg2 = getAveragePixel(centers_pixels_list2)#debug

    #debug
    # print "list 1 ", centers_pixels_list1, "min1 " , min1, "max1 " , max1 , "avg1 ", avg1
    # print "list 2 ", centers_pixels_list2, "min2 " , min2, "max2 " , max2 , "avg2 ", avg2

    #ci.drawCorners(initial_img, centers_points) # for debug
    #ci.show(initial_img) # for debug

    # if min1 < min2 
    if comparePixels(min1, min2) == 1:
        # min1 = black , min2 = white
        return [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)], [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)]
    else:
        # min1 = white, min2 = black
        first = 1
        return [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)], [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)]


################################################
#
def setRGBRanges(corners, img):

    centers_pixels_list1 = []
    centers_pixels_list2 = []

    # points for debug only
    centers_points = []
    k = 0
    flag_row = 1
    # loop over Chess board slots and get max and min pixel values for brown and white slots
    for i in range(8):
        flag_col = 1
        for j in range(8):
            # print k, k+1, k+9, k+10
            tl = corners[k][0]
            tr = corners[k+1][0]
            bl = corners[k+9][0]
            br = corners[k+10][0]
            # print tl, tr, bl, br
            dom = getDominantColor(tl, br, tr, bl, img)
            if flag_row:
                if flag_col: # white
                    centers_pixels_list2.append(dom)
                else: # brown
                    centers_pixels_list1.append(dom)
            else:
                if flag_col: # brown
                    centers_pixels_list1.append(dom)
                else: # white
                    centers_pixels_list2.append(dom)
            flag_col = 1-flag_col
            k += 1
        flag_row = 1-flag_row 
        k += 1

    # perform noise filtering of unrelevant pixels
    min1, max1 = getMinMaxFilteredPixel(centers_pixels_list1)
    min2, max2 = getMinMaxFilteredPixel(centers_pixels_list2)

    avg1 = getAveragePixel(centers_pixels_list1)#debug
    avg2 = getAveragePixel(centers_pixels_list2)#debug

    #debug
    # print "list 1 ", centers_pixels_list1, "min1 " , min1, "max1 " , max1 , "avg1 ", avg1
    # print "list 2 ", centers_pixels_list2, "min2 " , min2, "max2 " , max2 , "avg2 ", avg2

    # ci.drawCorners1(img, centers_points) # for debug
    # ci.show(img) # for debug

    # if min1 < min2 
    if comparePixels(min1, min2) == 1:
        # min1 = brown , min2 = white and first square is white
        first = 0
        return [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)], [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)], first
    else:
        # min1 = white, min2 = brown and first square is brown
        first = 1
        return [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)], [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)], first

########################################3

#
def CannyConfidence(tl, br, tr, bl, h, k, edges_img):
    points = getDiagonalsPoints(tl, br, tr, bl, h, k)
    samplesize = (k-1)*4
    length = len(points)
    for i in range(0, length):
        p = points[i]
        p = (int(p[0]), int(p[1]))
        rgb = edges_img[p[1], p[0]]
        points[i] = rgb
    w_count = points.count(255)
    return float(w_count)/samplesize

def isSquareEmpty(tl, br, tr, bl, edges_img):
    h = 10
    hinc = 0
    k = 100
    kfactor = 2
    threshO = 0.07
    threshV = 0.060
    maxsteps = 4
    ostepfactor = 1
    vstepfactor = 0
    threshstep = (threshO-threshV)/(maxsteps*(ostepfactor+vstepfactor))
    while True:
        con = CannyConfidence(tl,br,tr,bl,h,k,edges_img)
        if con > threshO:
            return 0
        if con < threshV:
            return 1
        threshO-=ostepfactor*threshstep
        threshV+=vstepfactor*threshstep
        k*=kfactor
        h+=hinc
#
def getMiddlePoint(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0])

#
def getDiagonalsByCorners(tl, br, tr, bl):
    yield (tl, br)
    yield (tr, bl)
    yield (getMiddlePoint(tl, tr), getMiddlePoint(bl, br))
    yield (getMiddlePoint(tl, bl), getMiddlePoint(tr, br))

# K = p1----p_cross , L = p_cross----p2
def getCrossPoint(p1, p2, K, L):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    den = (L + K)*1.0
    return np.array([(x1*L + x2*K)/den, (y1*L + y2*K)/den])

# dyagonal is devided to h parts, skip 1/h from each side
# each dyagonal is devided to k parts - take parts edges as inner points
def getDiagonalsPoints(tl, br, tr, bl, h, k):
    points_list = []
    for (p1, p2) in getDiagonalsByCorners(tl, br, tr, bl):
        length = np.linalg.norm(p1-p2)
        # we want to get p1 and p2 more close to center
        part_to_skip = length/h
        p1 = getCrossPoint(p1, p2, part_to_skip, length-part_to_skip)
        p2 = getCrossPoint(p1, p2, length-part_to_skip, part_to_skip)
        # calculate new length
        length = np.linalg.norm(p1-p2)
        interval = (length)/k
        for i in range(1, k):
            part = i*interval
            points_list.append(getCrossPoint(p1, p2, part, length-part))
    return points_list

#
def getDominantFromPoints(points, img):
    pixels_list = []
    for p in points:
        p = (int(p[0]), int(p[1]))
        #print "point - pixel", p, game_img[p[1], p[0]]
        pixels_list.append(img[p[1], p[0]])
    pixels_list = getFilteredList(pixels_list, 2)
    # cluster the pixel intensities
    clt = KMeans(n_clusters = 1)
    clt.fit(np.array(pixels_list))
    # find the most frequent color in current position
    counts = np.bincount(clt.labels_)
    freq_label = np.argmax(counts)
    # bgr color
    freq_color = clt.cluster_centers_[freq_label]
    # print freq_color
    return (int(freq_color[2]), int(freq_color[1]), int(freq_color[0]))

#
def getDominantColor(tl, br, tr, bl, img):
    points = getDiagonalsPoints(tl, br, tr, bl, 3, 50)
    #ci.drawCorners1(game_img, points)
    return getDominantFromPoints(points, img)


# side = 0, 1, 2, 3 to sign half-diagonal corner, 0=tl,1=tr,2=br,3=bl
def getHalfDiagonals(tl, br, tr, bl, side):
    center = getMiddlePoint(tl, br)
    middle_tl_tr = getMiddlePoint(tl, tr)
    middle_tl_bl = getMiddlePoint(tl, bl)
    middle_tr_br = getMiddlePoint(tr, br)
    middle_bl_br = getMiddlePoint(bl, br)
    return {
        0: [(tl, center), (middle_tl_tr, center), (middle_tl_bl, center)],
        1: [(tr, center), (middle_tl_tr, center), (middle_tr_br, center)],
        2: [(br, center), (middle_tr_br, center), (middle_bl_br, center)],
        3: [(bl, center), (middle_bl_br, center), (middle_tl_bl, center)],
    }[side]


# 
# each half dyagonal is devided to k parts - take parts edges as inner points
def getHalfDiagPoints(half_diag, k):
    points_list = []
    for (p1, p2) in half_diag:
        length = np.linalg.norm(p1-p2)
        # we want to get p1 more close to center
        part_to_skip = length/3
        p1 = getCrossPoint(p1, p2, part_to_skip, length-part_to_skip)
        # calculate new length
        length = np.linalg.norm(p1-p2)
        interval = (length)/k
        for i in range(1, k):
            part = i*interval
            points_list.append(getCrossPoint(p1, p2, part, length-part))
    return points_list

#
def getHalfDominants(tl, br, tr, bl, game_img):
    dominants = []
    for side in [0, 1, 2, 3]:
        half_diag = getHalfDiagonals(tl, br, tr, bl, side)
        points_list = getHalfDiagPoints(half_diag, 25)
        # ci.drawCorners1(game_img, points_list)#
        dom = getDominantFromPoints(points_list, game_img)
        print "side", side, "dom", dom #
        dominants.append(dom)
    return dominants


# get RGB color and return True if it's in range of range_color and False otherwise
def isColorInRange(range_color, rgb_color):
    return all(start <= color <= end for color, start, end in zip(rgb_color, range_color[0], range_color[1]))

#
def colorDiffFromRange(range_color, rgb_color):
    rgb = np.array(rgb_color)
    avg_range = getAveragePixel(range_color)
    rng = np.array(avg_range)
    return np.linalg.norm(rgb - rng)

# status : 0 for empty, 1 or 2 for occupied and None for unrecognized
# square color, 0 for white, 1 for brown
def getPositionStatusByDominantColor(dom_color, square_color):
    # if canny has no white on , return by square

    print "dom_color", dom_color
    # square is white and piece is black
    if square_color == 0 and isColorInRange(FIRST_PIECE_RGB, dom_color):
        return 1

    # square is brown and piece is white
    if square_color == 1 and isColorInRange(SECOND_PIECE_RGB, dom_color):
        return 2

    diff_list = []
    diff_piece1, diff_piece2, diff_brown, diff_white = -1, -1, -1, -1

    diff_piece1 = colorDiffFromRange(FIRST_PIECE_RGB, dom_color)
    diff_list.append(diff_piece1)

    diff_piece2 = colorDiffFromRange(SECOND_PIECE_RGB, dom_color)
    diff_list.append(diff_piece2)

    diff_brown = colorDiffFromRange(BROWN_RGB, dom_color)
    diff_list.append(diff_brown)

    diff_white = colorDiffFromRange(WHITE_RGB, dom_color)
    diff_list.append(diff_white)

    min_diff = min(diff_list)

    if (min_diff == diff_piece1 and isColorInRange(FIRST_PIECE_RGB, dom_color)) or (min_diff == diff_brown and square_color == 0):
        return 1
    if (min_diff == diff_piece2 and isColorInRange(SECOND_PIECE_RGB, dom_color)) or (min_diff == diff_white and square_color == 1):
        return 2
    if (min_diff == diff_brown and  square_color == 1) or (min_diff == diff_white and square_color == 0):
        return 0

    print "None", dom_color
    print "min_diff", min_diff
    print "diff_piece1", diff_piece1, "diff_piece2", diff_piece2, "diff_brown", diff_brown, "diff_white", diff_white
    return None
 

# square color, 0 for white, 1 for brown
def getPositionStatusByHalfDom(tl, br, tr, bl, game_img, square_color):
    new_dom_list = getHalfDominants(tl, br, tr, bl, game_img)
    print "none half dom" , new_dom_list #
    status_list = []
    for dom in new_dom_list:
        status = getPositionStatusByDominantColor(dom, square_color)
        print "dom", dom, "res", status
        status_list.append(status)
    if 1 in status_list and square_color == 0:
        return 1
    if 2 in status_list and square_color == 1:
        return 2
    if status_list.count(1)>=2: #
        return 1
    if status_list.count(2)>=2: #
        return 2
    if 1 in status_list and 2 not in status_list: #
        return 1
    if 2 in status_list and 1 not in status_list: #
        return 2
    return 0
 
#######################################################
