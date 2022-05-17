import numpy as np
import argparse
import cv2
import sys

# get what id to generate
ap = argparse.ArgumentParser()
ap.add_argument('-o', '--output', required=True)
ap.add_argument('-i', '--id', type=int, required=True)
ap.add_argument('-t', '--type', type=str, required=True)
args = vars(ap.parse_args())

# load dictionary
ARUCO_DICT = {
	'DICT_4X4_50': cv2.aruco.DICT_4X4_50,
	'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
	'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
	'DICT_4X4_1000': cv2.aruco.DICT_4X4_1000,
	'DICT_5X5_50': cv2.aruco.DICT_5X5_50,
	'DICT_5X5_100': cv2.aruco.DICT_5X5_100,
	'DICT_5X5_250': cv2.aruco.DICT_5X5_250,
	'DICT_5X5_1000': cv2.aruco.DICT_5X5_1000,
	'DICT_6X6_50': cv2.aruco.DICT_6X6_50,
	'DICT_6X6_100': cv2.aruco.DICT_6X6_100,
	'DICT_6X6_250': cv2.aruco.DICT_6X6_250,
	'DICT_6X6_1000': cv2.aruco.DICT_6X6_1000,
	'DICT_7X7_50': cv2.aruco.DICT_7X7_50,
	'DICT_7X7_100': cv2.aruco.DICT_7X7_100,
	'DICT_7X7_250': cv2.aruco.DICT_7X7_250,
	'DICT_7X7_1000': cv2.aruco.DICT_7X7_1000,
	'DICT_ARUCO_ORIGINAL': cv2.aruco.DICT_ARUCO_ORIGINAL,
	'DICT_APRILTAG_16h5': cv2.aruco.DICT_APRILTAG_16h5,
	'DICT_APRILTAG_25h9': cv2.aruco.DICT_APRILTAG_25h9,
	'DICT_APRILTAG_36h10': cv2.aruco.DICT_APRILTAG_36h10,
	'DICT_APRILTAG_36h11': cv2.aruco.DICT_APRILTAG_36h11
}

# get the dictionary
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args['type']])

# draw the id
print('Type %s \nID %i' % (args['type'], args['id']))
tag = np.zeros((300, 300, 1), dtype='uint8')
cv2.aruco.drawMarker(arucoDict, args['id'], 300, tag, 1)

# saving the id
cv2.imwrite(args['output'], tag)
cv2.imshow('ArUCo Tag', tag)
cv2.waitKey(0)