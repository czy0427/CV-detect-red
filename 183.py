# import numpy as np
# import cv2
#
# image = cv2.imread('red.jpg')
# #print(image)
# cv2.imshow('test', image)
# cv2.waitKey()
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
# fgbg = cv2.createBackgroundSubtractorMOG2();
# fgmask = fgbg.apply(image);
# cv2.imshow('GMG noise', fgmask);
# cv2.waitKey()
# fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);
# cv2.imshow('GMG', fgmask);
# cv2.waitKey()



# import numpy as np
# import cv2
#
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3));
#
# # creating object
# fgbg = cv2.bgsegm.createBackgroundSubtractorGMG();
#
# # capture frames from a camera
# cap = cv2.VideoCapture(0);
# while(1):
#     # read frames
#     ret, img = cap.read();
#
#     # apply mask for background subtraction
#     fgmask = fgbg.apply(img);
#
#     # with noise frame
#     cv2.imshow('GMG noise', fgmask);
#
#     # apply transformation to remove noise
#     fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel);
#
#     # after removing noise
#     cv2.imshow('GMG', fgmask);
#
#     k = cv2.waitKey(30) & 0xff;
#
#     if k == 27:
#         break;
#
# cap.release();
# cv2.destroyAllWindows();

####################################################
#
# from __future__ import print_function
# import cv2 as cv
# import argparse
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# #parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()
#
# #capture = cv.VideoCapture('test.mp4')
# #capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
# capture = cv.VideoCapture(0)
#
# if not capture.isOpened():
#     print('Unable to open: ' + args.input)
#     exit(0)
#
# time=0
# while True:
#     ret, frame = capture.read()
#     if frame is None:
#         break
#
#     fgMask = backSub.apply(frame)
#
#
#     cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
#     cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
#                cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
#
#
#     cv.imshow('Frame', frame)
#     cv.imshow('FG Mask', fgMask)
#
# #COUNT NUMBER OF 255
#     #print(fgMask[:,0])
#     #fgMask.shape=(1080,1920)
#     # initializing K
#     K = 200
#     cnt = 0
#     consec = False
#     for i in fgMask[:,0]:
#         if i == 255:
#             consec = True
#             cnt += 1
#         if consec == True and i==0:
#             cnt = 0
#             consec = False
#         if cnt == K:
#             time += 1
#             print("DETECTED!", time)
#             #img_name = "capture_{}.png".format(time)
#             #cv.imwrite(img_name, frame)
#
#
#     keyboard = cv.waitKey(30)
#     if keyboard == 'q' or keyboard == 27:
#         break
####################################################

# import numpy as np
# import cv2
#
# # Parameters
# blur = 21
# canny_low = 15
# canny_high = 150
# min_area = 0.0005
# max_area = 0.95
# dilate_iter = 10
# erode_iter = 10
# mask_color = (0.0,0.0,0.0)
#
# # initialize video from the webcam
# video = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = video.read()
#
#     if ret == True:
#         # Convert image to grayscale
#         image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # Apply Canny Edge Dection
#         edges = cv2.Canny(image_gray, canny_low, canny_high)
#         edges = cv2.dilate(edges, None)
#         edges = cv2.erode(edges, None)
#         # get the contours and their areas
#         contour_info = [(c, cv2.contourArea(c),) for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1]]
#         # Get the area of the image as a comparison
#         image_area = frame.shape[0] * frame.shape[1]
#
#         # calculate max and min areas in terms of pixels
#         max_area = max_area * image_area
#         min_area = min_area * image_area
#         # Set up mask with a matrix of 0's
#         mask = np.zeros(edges.shape, dtype = np.uint8)
#         # Go through and find relevant contours and apply to mask
#         for contour in contour_info:
#             # Instead of worrying about all the smaller contours, if the area is smaller than the min, the loop will break
#             if contour[1] > min_area and contour[1] < max_area:
#                 # Add contour to mask
#                 mask = cv2.fillConvexPoly(mask, contour[0], (255))
#         # use dilate, erode, and blur to smooth out the mask
#         mask = cv2.dilate(mask, None, iterations=mask_dilate_iter)
#         mask = cv2.erode(mask, None, iterations=mask_erode_iter)
#         mask = cv2.GaussianBlur(mask, (blur, blur), 0)
#         # Ensures data types match up
#         mask_stack = mask_stack.astype('float32') / 255.0
#         frame = frame.astype('float32') / 255.0
#         # Blend the image and the mask
#         masked = (mask_stack * frame) + ((1-mask_stack) * mask_color)
#         masked = (masked * 255).astype('uint8')
#         cv2.imshow("Foreground", masked)
#
#         # Use the q button to quit the operation
#         if cv2.waitKey(60) & 0xff == ord('q'):
#             break
#     else:
#         break
# cv2.destroyAllWindows()
# video.release()


###################################################################################
# import cv2
# import numpy as np
# table = cv2.imread('table.png')
# table_hsv = cv2.cvtColor(table, cv2.COLOR_BGR2HSV)
# #table_h = table_hsv[:, :, 0].flatten()
# #print(min(table_h), max(table_h))
#
# panda = cv2.imread('phone2.jpeg')
# #panda = cv2.imread('2obj.png')
# #panda = cv2.imread('key2.png')
# #panda = cv2.imread('panda2.png')
# panda_hsv = cv2.cvtColor(panda, cv2.COLOR_BGR2HSV)
# #print(panda_hsv)
# #panda_h = panda_hsv[:, :, 0].flatten()
# #print(min(panda_h), max(panda_h))
#
# panda_threshold = cv2.bitwise_not(cv2.inRange(panda_hsv, (10, 0, 0), (35, 255, 255))) #14,22 => 10,50
# contours, _ = cv2.findContours(panda_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #print(contours)
# #cv2.drawContours(panda, contours, -1, (0,255,0), 3)
#
# for cnt in contours:
#     (x, y, w, h) = cv2.boundingRect(cnt)
#
#     # epsilon = 0.1*cv2.arcLength(cnt,True)
#     # approx = cv2.approxPolyDP(cnt,epsilon,True)
#     # cv2.drawContours(panda, approx, -1, (0,255,0), 3)
#
#     area = cv2.contourArea(cnt)
#     if area > 400:
#         # epsilon = 0.1*cv2.arcLength(cnt,True)
#         # approx = cv2.approxPolyDP(cnt,epsilon,True)
#
#         # ellipse = cv2.fitEllipse(cnt)
#         # cv2.ellipse(panda,ellipse,(0,255,0),2)
#
#         #cv2.drawContours(panda, cnt, -1, (0,255,0), 3)
#         cv2.rectangle(panda, (x, y), (x + w, y + h), (0, 0, 255), 2)
#
#         # rect = cv2.minAreaRect(cnt)
#         # box = cv2.boxPoints(rect)
#         # box = np.int0(box)
#         # #print(box)
#         # cv2.drawContours(panda,[box],0,(0,0,255),2)
#
# #         print(cnt[0][0])
# #         print(cnt.shape)
#         #print(cnt)
#
#         # for i in cnt:
#         #     print(i[0])
#         #     print(panda.shape)
#         #     print("panda", panda[i[0][0],i[0][1],:])
#         (x1, y1, w1, h1) = cv2.boundingRect(cnt)
#         # print(x1)
#         # print(y1)
#         # print(w1)
#         # print(h1)
#         # print(panda.shape)
#         num=0
#         for j in range(x1,x1+w1):
#             for i in range(y1,y1+h1):
#                 if (panda_hsv[i,j,0]>=0 and panda_hsv[i,j,0]<=10) or (panda_hsv[i,j,0]>=160 and panda_hsv[i,j,0]<=180):
#                     num += 1
#                 #print(panda[i][j])
#         print(num/(w1*h1))
#
# #cv2.drawContours(panda, contours, -1, (0,255,0), 3)
#
# while True:
#     #cv2.imshow("tab", table)
#     cv2.imshow("pand", panda)
#     cv2.imshow("thresh", panda_threshold)
#     key = cv2.waitKey(30)
#     if key == ord('q'):
#         break

import cv2
import numpy as np
#cap = cv2.VideoCapture(0)
# time=0
# while True:
#     _, frame = cap.read()
#     cv2.imshow("frame", frame)
#
#     panda_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     panda_threshold = cv2.bitwise_not(cv2.inRange(panda_hsv, (10, 0, 0), (35, 255, 255)))
#     cv2.imshow("thresh", panda_threshold)
# #COUNT NUMBER OF 255
#     #print(fgMask[:,0])
#     #fgMask.shape=(1080,1920)
#     # initializing K
#     K = 800
#     cnt = 0
#     consec = False
#     for i in panda_threshold[:,0]:
#         if i == 255:
#             consec = True
#             cnt += 1
#         if consec == True and i==0:
#             cnt = 0
#             consec = False
#         if cnt > K:
#             time += 1
#             print("DETECTED!", time)
#             #img_name = "capture_{}.png".format(time)
#             #cv.imwrite(img_name, frame)
#
#    # panda = cv2.imread(frame)
#    #  panda_hsv = cv2.cvtColor(panda, cv2.COLOR_BGR2HSV)
#    #  panda_threshold = cv2.bitwise_not(cv2.inRange(panda_hsv, (10, 0, 0), (35, 255, 255))) #14,22 => 10,50
#    #  contours, _ = cv2.findContours(panda_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     key = cv2.waitKey(30)
#     if key == ord('p'):
#         break
# cap.release()
# cv2.destroyAllWindows()

vc = cv2.VideoCapture(1)

if vc.isOpened():
    rval, image = vc.read()
else:
    rval = False

rval, image = vc.read()
while rval:
    cv2.imshow("feed", image)
# while True:
#     _, image = cap.read()q
#     cv2.imshow("feed", image)

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_threshold = cv2.bitwise_not(cv2.inRange(image_hsv, (10, 0, 0), (35, 255, 255)))  # 14,22 => 10,50
    contours, _ = cv2.findContours(image_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > 400:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.imshow("feed", image)

            # Count red pixels
            red_ct = 0
            for j in range(x, x+w, 5):
                for i in range(y, y+h, 5):
                    if (image_hsv[i, j, 0] >= 0 and image_hsv[i, j, 0] <= 10) or (image_hsv[i, j, 0] >= 160 and image_hsv[i, j, 0] <= 180):
                        red_ct += 1
            if red_ct / (w*h/25) > 0.7:
                print("RED")
            # else:
            #     print("NOT RED")

    # Read webcam feed
    rval, image = vc.read()
    # Exit if 'q' pressed
    key = cv2.waitKey(30)
    if key == ord('q'):
        break


'''
wall = cv2.imread("r?.png")
wall_hsv = cv2.cvtColor(wall, cv2.COLOR_BGR2HSV)

wall_h, wall_s, wall_v = wall_hsv[:,:,0].flatten(), wall_hsv[:,:,1].flatten(), wall_hsv[:,:,2].flatten()
print("mins:", min(wall_h), min(wall_s), min(wall_v), "maxs:", max(wall_h), max(wall_s), max(wall_v))
#mins: 11 32 102 maxs: 24 73 206

'''
