import cv2
import numpy as np
import time
import imutil
import matplotlib.pyplot as plt
import head
import argparse as ap

parse = ap.ArgumentParser()
parse.add_argument("-i", "--input", required=True, help="Path to the video")

args = vars(parse.parse_args())

# intinialise opencv for video capture
cam = cv2.VideoCapture(args['input'])

# enable aruco id tracking library

# i am using the 5x5 dictionary with 50 identical ids
# it is the smallest library so the most efficient
aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# create id detector
param = cv2.aruco.DetectorParameters_create()

# list of locations for calculating the velocity
loc_5 = []
loc_39 = []

# list of velocities 
v_5 = []
v_39 = []

# list of frames at which the velocities are sampled
f_5 = []
f_39 = []

# timer for frame rate calculaion
pTime = 0

# processing_frame rate
fps = 0

# current frame number
f_cnt = 0

# video frame rate
v_fps = 240

# function used to calculate velocity
def calcV(id, cX, cY):
    # check if it is first sample to prevent crashing
    if id == 5 and len(loc_5) > 0:
        # get previous sample's location
        x = loc_5[-1][0]
        y = loc_5[-1][1]

        # get previous frame
        f = loc_5[-1][2]

        # calculate velocity using last location and last frame
        # multiplying by fps of 240
        v = np.sqrt(((cX - x) ** 2 + (cY-y) ** 2)) / (f_cnt - f) * v_fps
        return v
    
    elif id == 39 and len(loc_39) > 0:
        # get previous sample's location
        x = loc_39[-1][0]
        y = loc_39[-1][1]

        # get previous frame
        f = loc_39[-1][2]

        # calculate velocity using last location and last frame
        # multiplying by fps of 240
        v = np.sqrt(((cX - x) ** 2 + (cY-y) ** 2)) / (f_cnt - f) * v_fps
        return v

    return 0

while True:
     # read webcam
    ret_val, img = cam.read()

    ###***  REMOVED BECAUSE SMALL RESOLUTION CAUSES LESS ACCURATE DATA  ***###
    # resize image to reduce CPU load                                        #
    # img = imutil.resize(img, 600)                                          #
    ##########################################################################

    # look for marker in image
    (corners, ids, rejected) = cv2.aruco.detectMarkers(img, aruco, parameters=param)

    # frame counter
    f_cnt += 1
    
    # if markers detected
    if len(corners) > 0:
        print('detected')

        # resize array
        ids = ids.flatten()

        # iterate through all the ids found
        for (corner, id) in zip(corners, ids):
            # resize array
            corners_abcd = corner.reshape((4,2))

            # getting four corner of id's bounding box
            (tl, tr, br, bl) = corners_abcd
            tr_pnt = [int(tr[0]), int(tr[1])]
            tl_pnt = [int(tl[0]), int(tl[1])]
            br_pnt = [int(br[0]), int(br[1])]
            bl_pnt = [int(bl[0]), int(bl[1])]

            # draw the polygon
            enclose = np.array([[tr_pnt, tl_pnt, bl_pnt, br_pnt]], np.int32)
            cv2.polylines(img, [enclose], True, (0,255,0), thickness=3)

            # get the location of center of the id
            cX = (tl[0]+br[0]) / 2
            cY = (tl[1]+br[1]) / 2
            dx = int(cX)
            dy = int(cY)
            cv2.circle(img, (dx, dy), 4, (255, 0, 0), -1)

            # calculate velocity using calcV() function
            v = calcV(id, cX, cY)
            print('[Marker %i]: (%i, %i) @ %f' % (id, cX, cY, v))

            # saving data for plotting
            if id == 5:
                loc_5.append((cX, cY, f_cnt))
                v_5.append(v)
                f_5.append(f_cnt)

            elif id == 39:
                loc_39.append((cX, cY, f_cnt))
                v_39.append(v)
                f_39.append(f_cnt)

            # visualisation on page
            cv2.putText(img, 'id%i @ %.3fpix/sec' % (id, v), (int(tl[0]-10),   int(tl[1]-10)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))



    # frame rate calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, 'FPS: %d | Frame: %d' % (fps, f_cnt), (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)

    # show image
    cv2.imshow('Tracking', img)

    # esc to quit
    if cv2.waitKey(1) == 27: 
        break  

# get the first and last frame one of the ids is detected for drawing the graph
if len(f_5) > 0 and len(f_39) > 0:
    f_init = f_5[0] if f_5[0] < f_39[0] else f_39[0]
    f_final = f_5[-1] if f_5[-1] > f_39[-1] else f_39[-1]

else:
    f_init = f_5[0] if len(f_5) > 0 else f_39[0]
    f_final = f_5[-1] if len(f_5) > 0 else f_39[-1]

# calculating line of best fit using numpy
bf_5 = np.polyfit(f_5, v_5, 1)
l_5 = bf_5[1] + np.multiply(bf_5[0], f_5)

# comment out next two lines for single cart tracking
bf_39 = np.polyfit(f_39, v_39, 1)
l_39 = bf_39[1] + np.multiply(bf_39[0], f_39)

# close any previous graphing window
plt.close('all')

# first graph for cart with ID 5
plt.subplot(2,1,1)
plt.title('Draft Launch')
plt.plot(f_5, v_5, 'o')
plt.plot(f_5, l_5, 'r', label='ID 5 -- y = %.7fx + %.3f' % (bf_5[0], bf_5[1]))
plt.grid()
plt.legend()
plt.ylabel('pixels/sec') 
plt.xlim((f_init, f_final))
plt.xticks(np.arange(f_init, f_final, step=30))

# second graph for cart with ID 39 - comment out the following block for single cart tracking
plt.subplot(2,1,2)
plt.plot(f_39, v_39, 'o') 
plt.plot(f_39, l_39, 'g', label='ID 39 -- y = %.7fx + %.3f' % (bf_39[0], bf_39[1]))
plt.grid() 
plt.legend()
plt.xlabel('Frame')
plt.ylabel('pixels/sec') 
plt.xlim((f_init, f_final))
plt.xticks(np.arange(f_init, f_final, step=30))


plt.show()

head.st()

print('goodnight')