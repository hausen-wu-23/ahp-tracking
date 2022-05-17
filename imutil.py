# importing libraries
import numpy as np
import cv2

# translate is used for shifting entire image by x and y amounts
def translate(image, x, y):
    # translation matrix
    M = np.float32([[1, 0, x], [0, 1, y]])
    
    # applying the matrix
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return(shifted)

# rotate is used for rotating entire image by {angle} degrees around the {ctr} 
def rotate(img, angle, ctr=None, scale=1.0):
    # get shape
    (h, w) = img.shape[:2]

    # if no center specified
    if ctr is None:
        ctr = (w//2, h//2)

    # rotation matrix
    M = cv2.getRotationMatrix2D(ctr, angle, scale)

    # apply matrix
    rotated = cv2.warpAffine(img, M, (w,h))

    return rotated

# resize is used for scaling the image
def resize(img, width=None, height=None, alg=cv2.INTER_AREA):
    (h, w) = img.shape[:2]

    # return excpetion if no arguments given
    if width is None and height is None:
        raise Exception('Both width and height are empty')

    # if no width given, rescale with height
    if width is None:
        ratio = height / float(h)
        dimension = (int(w * ratio), height)
    
    # if no height given, rescale with width
    elif height is None:
        ratio = width / float(w)
        dimension = (width, int(h * ratio))

    # rescale with width and height given
    else:
        dimension = (width, height)

    # resize the image and return
    resized = cv2.resize(img, dimension, interpolation=alg)
    return resized