import numpy as np
import cv2
num = 7

# get board image, return inner corners
def getInnerCorners(img):
    ret, corners = cv2.findChessboardCorners(img, (num,num), None)
    return corners

# get board image in gray color and the origin image, 
def undistort(img,orig):
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((num*num,3), np.float32)
    objp[:,:2] = np.mgrid[0:num,0:num].T.reshape(-1,2)
    corners = getInnerCorners(img)
    if corners is None:
        return None
    cv2.cornerSubPix(img,corners,(8,8),(-1,-1),criteria)
    calib = cv2.calibrateCamera([objp],[corners],img.shape[::-1],None,None)
    ret, mtx, dist, rvecs, tvecs = calib
    h, w = img.shape[:2]
    cameramatx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1.0,(w,h))
    ret = cv2.undistort(orig,mtx,dist,None,cameramatx)
    #x,y,w,h = roi
    #ret = ret[y:y+h, x:x+w]
    return ret
# 
def extrapolatepoint(corners, ind0, ind1, ind2):
    return 3*corners[ind0] - 3*corners[ind1] + corners[ind2]

#
def extrapolateOuterCorners(innercorners,length = 7):
    innercorners = innercorners
    ret = []
    zerocorner = extrapolatepoint(innercorners,0,length+1, length*2+2)
    ret.append(zerocorner)
    for i in xrange(length):
        ret.append(extrapolatepoint(innercorners,i,i+length,i+2*length))
    eightcorner = extrapolatepoint(innercorners,length-1,(length-1)*2,(length-1)*3)
    ret.append(eightcorner)
    for i in xrange(length):
        ret.append(extrapolatepoint(innercorners,length*i,length*i+1,length*i+2))
        for j in xrange(length):
            ret.append(innercorners[length*i+j])
        ret.append(extrapolatepoint(innercorners,length*(i+1)-1,length*(i+1)-2,length*(i+1)-3))
    fivesixcorner = extrapolatepoint(innercorners,length*(length-1),(length-1)**2, length**2-3*length+2)
    ret.append(fivesixcorner)
    for i in xrange(length):
        ret.append(extrapolatepoint(innercorners, length * (length-1) + i, length * (length - 2) + i, length * (length - 3) + i))
    sixfourcorner = extrapolatepoint(innercorners, length ** 2 - 1, length ** 2 - length - 2,length ** 2 - 2*length - 3)
    ret.append(sixfourcorner)
    return ret

#
def drawCorners(orig, corners):
    i = 0
    textscale = 5
    for corner in corners:
        corner = corner[0]
        corner = (int(corner[0]), int(corner[1]))
        cv2.circle(orig,corner,2*textscale,(255,0,0),-1)
        corner = (int(corner[0]-textscale),int(corner[1]+textscale))
        cv2.putText(orig,str(i),corner,cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
        i+=1
    return orig

def drawCorners1(orig, corners):
    i = 0
    textscale = 5
    for corner in corners:
        corner = (int(corner[0]), int(corner[1]))
        cv2.circle(orig,corner,2*textscale,(255,0,0),-1)
        corner = (int(corner[0]-textscale),int(corner[1]+textscale))
        cv2.putText(orig,str(i),corner,cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
        i+=1
    return orig

#
def show(img):
    cv2.imshow("result",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#
def getRectangles(corners, length=9):
    for row in xrange(length-1):
        for col in xrange(length-1):
            indz = row*length + col
            indo = indz+1
            indn = (row+1)*length + col
            indt = indn+1
            tl = corners[indz]
            tr = corners[indo]
            bl = corners[indn]
            br = corners[indt]
            yield tl,br,tr,bl
