# define set of functions for manipulating points and colors
import cv2
import numpy as np
from sklearn.cluster import KMeans

FIRST_PIECE_RGB = [(0,0,0), (60,60,60)]
SECOND_PIECE_RGB = [(60,60,60), (180,180,180)]
CELL_RANGE_MATRIX = None

# get a list of pixels (list of (r,g,b) taples) and return the
# "minimum pixel" defined as a pixel that composed of minimum on
# R values, minimum on G values and minimum on B values
def getMinPixelFromList(pixels_list):
    R = []
    G = []
    B = []
    for pixel in pixels_list:
        R.append(pixel[0])
        G.append(pixel[1])
        B.append(pixel[2])
    return (min(R), min(G), min(B))

# get a list of pixels (list of (r,g,b) taples) and return the
# "maximum pixel" defined as a pixel that composed of maximum on
# R values, maximum on G values and maximum on B values
def getMaxPixelFromList(pixels_list):
    R = []
    G = []
    B = []
    for pixel in pixels_list:
        R.append(pixel[0])
        G.append(pixel[1])
        B.append(pixel[2])
    return (max(R), max(G), max(B))

# get a list of pixels (list of (r,g,b) taples) and return the
# "average pixel" defined as a pixel that composed of average on
# R values, average on G values and average on B values
def getAveragePixel(pixels_list):
    r, g, b = 0, 0, 0
    count = 0
    for pixel in pixels_list:
        r += pixel[0]
        g += pixel[1]
        b += pixel[2]
        count += 1
    return ((r/count), (g/count), (b/count))

# compare between two pixels p1 and p2 with the logic below:
# if p1<p2 return 1, if p1>p2 return 2, othewise return 0
# p1 < p2 if (p1's R value < p2's R value and p1's G value < p2's G value and
# p1's B value < p2's B value), and vice versa with p1 > p2
def comparePixels(p1, p2):
    if p1[0] < p2[0] and p1[1] < p2[1] and p1[2] < p2[2]:
        return 1
    if p1[0] > p2[0] and p1[1] > p2[1] and p1[2] > p2[2]:
        return 2
    return 0

# get a list of pixels (list of (r,g,b) taples) and return this list after
# filtering it of outliers defined as values that are lower than
# (1/epsilon)*average_pixel or higher than epsilon*average_pixel
def getFilteredList(pixels_list, epsilon):
    avg = getAveragePixel(pixels_list)
    # print pixels_list, "avg", avg
    tmp = 1/epsilon
    avg_min = (tmp*avg[0], tmp*avg[1], tmp*avg[2])
    avg_max = (epsilon*avg[0], epsilon*avg[1], epsilon*avg[2])
    pixels_list = [p for p in pixels_list if comparePixels(p, avg_min) != 1 and comparePixels(p, avg_max) != 2]
    # print "after filtered", pixels_list
    return pixels_list

# get min and max pixel from pixels list after filtered
def getMinMaxFilteredPixel(pixels_list):
    # pixels_list = getFilteredList(pixels_list, 3.0)
    min_pixel = getMinPixelFromList(pixels_list)
    max_pixel = getMaxPixelFromList(pixels_list)
    return min_pixel, max_pixel

# get an image of a board at the begining of the game, and resolve its direction :
# return 1 for horizontal board thus two top and bottom rows are placed with pieces at the begining
# return 0 for vertical board thus two right and left rows are placed with pieces at the begining
def resolveBoardDirection(corners, initial_img):
    edges_img = cv2.Canny(initial_img,70,110) #
    count_empty = 0
    for i in [3, 4]:
        # sample first and last columns
        for j in [0, 7]:
            h = i*9+j
            tl = corners[h][0]
            tr = corners[h+1][0]
            bl = corners[h+9][0]
            br = corners[h+10][0]
            if isSquareEmpty(tl, br, tr, bl, edges_img):
                count_empty = count_empty +1
    if count_empty > 2:
        return 1
    else:
        return 0

# get an image of a board at the begining of the game, and
# resolve the two pieces colors
# return pieces' black color range and white color range
def initialPiecesRGB(corners, initial_img):
    # sample colors in squares' centers of two rows in each
    # side of the initial board, to get pieces' two colors
    centers_pixels_list1 = []
    centers_pixels_list2 = []
    k = 0
    if resolveBoardDirection(corners, initial_img) == 1:
        # board is horizontal thus two top and bottom rows are placed with pieces at the begining

        # sample from two first rows
        for i in [0,1]:
            for j in range(8):
                k = i*9+j
                tl = corners[k][0]
                tr = corners[k+1][0]
                bl = corners[k+9][0]
                br = corners[k+10][0]
                col = getDominantColor(tl, br, tr, bl, initial_img)
                centers_pixels_list1.append(col)

        # increase k by 8 + 1 steps in each row till 6th row = 9*4
        k += 36
        # sample from two last rows
        for i in [6,7]:
            for j in range(8):
                k=i*9+j
                tl = corners[k][0]
                tr = corners[k+1][0]
                bl = corners[k+9][0]
                br = corners[k+10][0]
                col = getDominantColor(tl, br, tr, bl, initial_img)
                centers_pixels_list2.append(col)
    else:
        # board is vertical thus two right and left rows are placed with pieces at the begining

        # sample from two right columns
        for i in range(8):
            for j in [0,1]:
                h = j+i*9
                tl = corners[h][0]
                tr = corners[h+1][0]
                bl = corners[h+9][0]
                br = corners[h+10][0]
                col = getDominantColor(tl, br, tr, bl, initial_img)
                centers_pixels_list1.append(col)

        # sample from two left columns
        k = 0
        for i in range(8):
            for j in [6, 7]:
                h = i*9 +j
                tl = corners[h][0]
                tr = corners[h+1][0]
                bl = corners[h+9][0]
                br = corners[h+10][0]
                col = getDominantColor(tl, br, tr, bl, initial_img)
                centers_pixels_list2.append(col)

    # perform noise filtering of unrelevant pixels
    min1, max1 = getMinMaxFilteredPixel(centers_pixels_list1)
    min2, max2 = getMinMaxFilteredPixel(centers_pixels_list2)

    # if min1 < min2
    if comparePixels(min1, min2) == 1:
        # min1 = black , min2 = white
        return [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)], [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)]
    else:
        # min1 = white, min2 = black
        first = 1
        return [(min2[0]-10, min2[1]-10, min2[2]-10), (max2[0]+10, max2[1]+10, max2[2]+10)], [(min1[0]-10, min1[1]-10, min1[2]-10), (max1[0]+10, max1[1]+10, max1[2]+10)]

# get ratio of white edge pixels to black edge pixels from
# canny image for a cell
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

# return whether a cell defined by 4 corners is empty
def isSquareEmpty(tl, br, tr, bl, edges_img):
    h = 100
    hinc = 0
    k = 100
    kfactor = 2
    threshO = 0.07
    threshV = 0.060
    maxsteps = 4
    ostepfactor = 10
    vstepfactor = 1
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

# get the middle point between points p1 and p2
def getMiddlePoint(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    return np.array([(x1+x2)/2.0, (y1+y2)/2.0])

# return 4 diafonals of a cell defined of 4 corners
def getDiagonalsByCorners(tl, br, tr, bl):
    yield (tl, br)
    yield (tr, bl)
    yield (getMiddlePoint(tl, tr), getMiddlePoint(bl, br))
    yield (getMiddlePoint(tl, bl), getMiddlePoint(tr, br))

# calculate p_cross between p1 and p2 thus its diff from p1 is K and
# its diff from p2 is L
# K = p1----p_cross , L = p_cross----p2
def getCrossPoint(p1, p2, K, L):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    den = (L + K)*1.0
    return np.array([(x1*L + x2*K)/den, (y1*L + y2*K)/den])

# dyagonal is devided to h parts, skip 1/h from each side (to get points from the cell inside)
# each dyagonal is devided to k parts - take parts edges as inner points
# return a list of those points retrieved from the 4 cell's diagonals
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

# get a list of pixels (list of (r,g,b) taples) and return a dominant pixel
def getDominantFromPoints(points, img):
    pixels_list = []
    for p in points:
        p = (int(p[0]), int(p[1]))
        pixels_list.append(img[p[1], p[0]])
    #pixels_list = getFilteredList(pixels_list, 2.0)

    # cluster the pixel intensities
    clt = KMeans(n_clusters = 1)
    clt.fit(np.array(pixels_list))
    # find the most frequent color in current position
    counts = np.bincount(clt.labels_)
    freq_label = np.argmax(counts)
    # bgr color
    freq_color = clt.cluster_centers_[0]
    # print freq_color
    return (int(freq_color[2]), int(freq_color[1]), int(freq_color[0]))

# get a list of pixels (list of (r,g,b) taples) and return 2 dominant pixels
def get2DominantFromPoints(points,img):
    pixels_list = []
    for p in points:
        p = (int(p[0]), int(p[1]))
        #print "point - pixel", p, game_img[p[1], p[0]]
        pixels_list.append(img[p[1], p[0]])
    #pixels_list = getFilteredList(pixels_list, 2.0)
    #print len(pixels_list)
    # cluster the pixel intensities
    clt = KMeans(n_clusters = 2)
    clt.fit(np.array(pixels_list))
    freq_color = map(lambda x: (int(x[2]), int(x[1]), int(x[0])),clt.cluster_centers_)
    # print freq_color
    return freq_color

# return the dominant pixel of a cell
def getDominantColor(tl, br, tr, bl, img):
    points = getDiagonalsPoints(tl, br, tr, bl, 3.0, 50)
    #ci.drawCorners1(game_img, points)
    return getDominantFromPoints(points, img)

# return the 2 dominant pixels of a cell
def get2DominantColor(tl, br, tr, bl, img):
    points = getDiagonalsPoints(tl, br, tr, bl, 3.0, 50)
    #ci.drawCorners1(game_img, points)
    return get2DominantFromPoints(points, img)

# return 3 half diagonals from each corner of the cell
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

# each half dyagonal is devided to k parts - take parts edges as inner points
# return a list of those points retrieved from the cell's half diagonals
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

# return the 4 dominant pixels of a cell
def getHalfDominants(tl, br, tr, bl, game_img):
    dominants = []
    for side in [0, 1, 2, 3]:
        half_diag = getHalfDiagonals(tl, br, tr, bl, side)
        points_list = getHalfDiagPoints(half_diag, 25)
        dom = getDominantFromPoints(points_list, game_img)
        dominants.append(dom)
    return dominants


# get RGB color and return True if it's in range of range_color and False otherwise
def isColorInRange(range_color, rgb_color):
    return all(start <= color <= end for color, start, end in zip(rgb_color, range_color[0], range_color[1]))

# return the diff of a color from range's average
def colorDiffFromRange(range_color, rgb_color):
    rgb = np.array(rgb_color)
    avg_range = getAveragePixel(range_color)
    rng = np.array(avg_range)
    return np.linalg.norm(rgb - rng)

# retrun a colors range for a square = cell defined by its 4 corners
def RangeforSquare(tl, br, tr, bl, img):
    points = map(lambda x: img[int(x[1]),int(x[0])],getDiagonalsPoints(tl, br, tr, bl, 3, 50))
    min = getMinPixelFromList(points)
    max = getMaxPixelFromList(points)
    return [min,max]

# retrun matrix of the cells ranges
def SetCellRanges(corners,img):
    ret = [[None for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            k =i+j*9
            tl = corners[k][0]
            tr = corners[k+1][0]
            bl = corners[k+9][0]
            br = corners[k+10][0]
            ret[i][j] = RangeforSquare(tl,br,tr,bl,img)
    return ret

# get dominant color of a square and the square color = 0 for white, 1 for brown and
# the square colors range and whether this cell is recognized as empty by canny and
# return status : 0 for empty, 1 for black piece, 2 for white piece and
# None for unrecognized
def getPositionStatusByDominantColor(dom_color, square_color, square_range, empty):
    # if canny has no white on , return by square
    if empty:
        if isColorInRange(FIRST_PIECE_RGB, dom_color):
            return 1
        else:
            return 0

    # square is white and piece is black
    if square_color == 0 and isColorInRange(FIRST_PIECE_RGB, dom_color):
        return 1

    # square is brown and piece is white
    if square_color == 1 and isColorInRange(SECOND_PIECE_RGB, dom_color):
        return 2

    if isColorInRange(square_range, dom_color):
        return 0

    diff_piece1 = colorDiffFromRange(FIRST_PIECE_RGB, dom_color)
    diff_piece2 = colorDiffFromRange(SECOND_PIECE_RGB, dom_color)

    if diff_piece1 < diff_piece2:
        return 1
    if diff_piece2 < diff_piece1:
        return 2

    return None

# get list of 2 dominant colors of a square and the square color = 0 for white, 1 for brown and
# the square colors range and whether this cell is recognized as empty by canny and
# return status : 0 for empty, 1 for black piece, 2 for white piece and
# None for unrecognized
def getPositionStatusBy2DominantColor(dom_color, square_color, square_range, empty):
    # if canny has no white on , return by square
    diffs = map(lambda x: colorDiffFromRange(square_range,x), dom_color)
    if diffs[0] > diffs[1]:
        colorF , colorB = dom_color
    else:
        colorB, colorF = dom_color

    if empty:
        if isColorInRange(FIRST_PIECE_RGB, colorF):
            return 1
        #if square_color == 1 and isColorInRange(SECOND_PIECE_RGB, colorF):
        #    return 2
        return 0

    # square is white and piece is black
    if square_color == 0 and isColorInRange(FIRST_PIECE_RGB, colorF):
        return 1

    # square is brown and piece is white
    if square_color == 1 and isColorInRange(SECOND_PIECE_RGB, colorF):
        return 2

    if isColorInRange(square_range, colorB):
        return 0

    diff_piece1 = colorDiffFromRange(FIRST_PIECE_RGB, colorF)
    diff_piece2 = colorDiffFromRange(SECOND_PIECE_RGB, colorF)

    if diff_piece1 < diff_piece2:
        return 1
    if diff_piece2 < diff_piece1:
        return 2

    return None

# position status recognition step 2
# get half dominant color of a square and the square color = 0 for white, 1 for brown and
# the square colors range and whether this cell is recognized as empty by canny and
# return status : 0 for empty, 1 for black piece, 2 for white piece and
# None for unrecognized
def getPositionStatusByHalfDom(tl, br, tr, bl, game_img, square_color, square_range,empty):
    new_dom_list = getHalfDominants(tl, br, tr, bl, game_img)
    print "none half dom" , new_dom_list #
    status_list = []
    for dom in new_dom_list:
        status = getPositionStatusByDominantColor(dom, square_color, square_range,empty)
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

# return 0 for white piece, 1 for black piece and none for empty cell
def getResultForSquare(corners, k, edges_img, game_img, square_color, i, j):
    empty = 0
    tl = corners[k][0]
    tr = corners[k+1][0]
    bl = corners[k+9][0]
    br = corners[k+10][0]
    if isSquareEmpty(tl, br, tr, bl, edges_img):
        empty = 1
    dom_color = get2DominantColor(tl, br, tr, bl, game_img)
    result = getPositionStatusBy2DominantColor(dom_color, square_color, CELL_RANGE_MATRIX[i][j], empty)
    if result is None:
        result = getPositionStatusByHalfDom(tl, br, tr, bl, game_img, square_color, CELL_RANGE_MATRIX[i][j],empty)
    if result == 0 or result is None:
        return None
    else:
        return 2-result

# return a matrix of cells' statuses
def getResultsForOneFrame(game_img, edges_img, corners, square_color):
    result_matrix = np.empty([8, 8])
    k = 0
    # loop over Chess board squares and mark in matrix the squares' statuses (occupied or not)
    for i in range(8):
        for j in range(8):
            result_matrix[i][j] = getResultForSquare(corners,k,edges_img,game_img,square_color,i,j)
            square_color = 1-square_color
            k += 1
        square_color = 1-square_color
        k += 1

    return result_matrix

# compare between two rgb colors by its sums
def cmpColor(first,second):
    return cmp(sum(first),sum(second))

# return 0 if one range contains the other, 1 if first range > second and -1 otherwise
def cmpRanges(first,second):
    ret = cmpColor(first[0],second[0])+cmpColor(first[1],second[1])
    if ret == 0:
        return 0
    return 1 if ret > 0 else -1

# return the color of the first square in the board, 1 for brown cell and 0 for white one
def firstSquareColor():
    firstrange = CELL_RANGE_MATRIX[0][0]
    contenderindices = [2*i+(1 if i%8 < 4 else 0) for i in xrange(32)]
    contenders = [CELL_RANGE_MATRIX[i%8][i/8] for i in contenderindices]
    victories = 0 #times first was higher than contender vs the opposite
    for contender in contenders:
        victories+=cmpRanges(firstrange,contender)
    if victories > len(contenders)/2:
        return 0 #white square
    return 1 #brown square

#######################################################