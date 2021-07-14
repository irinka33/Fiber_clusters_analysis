# import the necessary packages
#from scipy.spatial import distance as dist
#from imutils import perspective
#from imutils import contours
import numpy as np
#import imutils
import math
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

#file_name = 'data\input\image1.jpg'

file_name = 'data\input\imageSS27.jpg'
contours_file = 'data\input\contoursSS27.cz'

#file_name = 'data\img_upper_part.jpg'
#file_name = 'data\img_lower_part.jpg'

#file_name = 'data\cell_test.jpg'

def open_contours(contours_file):
    contours = []

    tree = ET.parse(contours_file)
    root = tree.getroot()

    #for cntr in root.findall('country'):
    for cntr in root.iter('Bezier'):
        id = cntr.attrib['Id']
        points = cntr.find('Geometry').find('Points').text.split(' ')

        contour = []
        for point in points:
            pnt = [int(float(point.split(',')[0])), int(float(point.split(',')[1]))]
            contour.append([pnt])

        contours.append(np.array(contour))
        print(id)

    return contours

def draw_contours(image, contours):
    cv2.namedWindow("Output Contours", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.resizeWindow("Output Contours", 900, 600)

    cv2.drawContours(image, contours, -1, (0, 255, 0), 2) #green
    cv2.imshow("Output Contours", image)
    cv2.waitKey(0)

def draw_circles(image, contours):
    cv2.namedWindow("Circles", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.resizeWindow("Circles", 900, 600)

    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(image, center, radius, (0, 255, 0), 2) # green

        radius = int(math.sqrt(area/math.pi))
        cv2.circle(image,(cx, cy), radius, (255, 0, 0), 2) # blue
        cv2.circle(image,(cx, cy), 1, (255, 0, 0), 2)

    cv2.imshow("Circles", image)
    cv2.waitKey(0)

def canny_edge(image):

    #blur = cv2.GaussianBlur(image, (5, 5), 0)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray", imgray)
    cv2.waitKey(0)

    edges = cv2.Canny(imgray, 250, 254)

    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

def turning_image(image):

    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Gray", imgray)
    cv2.waitKey(0)

    blur = cv2.GaussianBlur(imgray, (5, 5), 0)
    cv2.imshow("Blur", blur)
    cv2.waitKey(0)

    ret, thresh = cv2.threshold(imgray, 100 , 255, cv2.THRESH_TRIANGLE)

    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    orig = image.copy()
    cv2.drawContours(orig, contours, -1, (0, 255, 0), 2)

    cv2.putText(orig, str(len(contours)),(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Contours", orig)
    cv2.waitKey(0)

    #img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    img = image.copy()
    #cv2.imshow("Circles", img)
    for cnt in contours:
        if cv2.contourArea(cnt) > 0:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(img, center, radius, (0, 255, 0), 2)
     #   cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
    cv2.imshow("Circles", img)
    cv2.waitKey(0)

    #cv2.circle(img, (500, 500), 100, (0, 0, 255), 2)
    #cv2.imshow("Circles", img)
    #cv2.waitKey(0)

def find_contuors(image):
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.GaussianBlur(gray, (7, 7), 0)

    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    #edged = cv2.Canny(gray, 250, 254)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)
    ret, edged = cv2.threshold(gray, 200 , 255, cv2.THRESH_BINARY)

    #cv2.imshow("Thresh", edged)
    #cv2.waitKey(0)

    edged = cv2.dilate(edged, None, iterations=3)
    #cv2.imshow("Dilate", edged)
    #cv2.waitKey(0)
    edged = cv2.erode(edged, None, iterations=1)
    #cv2.imshow("Erode", edged)
    #cv2.waitKey(0)
    # find contours in the edge map
    cnts, hir = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)

    # compute the rotated bounding box of the contour
    orig = image.copy()
    cv2.drawContours(orig, cnts, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", orig)
    cv2.waitKey(0)

    img = image.copy()
    #cv2.imshow("Circles", img)
    for cnt in cnts:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)

        if 200 <= area <= 15000:
            color = (255, 0, 0)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # skip "black" areas
            (b, g, r) = image[cy,cx]
            if (b <= 35) and (g <= 35) and (r <= 35):
                #continue
                color = (255, 255, 255)

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            area_circle = math.pi * radius * radius

            # skip 'not-febrile' areas using shape analysis
            if area_circle/(area+0.000001) >= 4:
                continue
                #color = (0, 0, 255)

            cv2.circle(img, center, radius, (0, 255, 0), 2)

            # cx = int(M['m10'] / M['m00'])
            # cy = int(M['m01'] / M['m00'])
            radius = int(math.sqrt(area/math.pi))
            cv2.circle(img,(cx, cy), radius, color, 2)
            cv2.circle(img,(cx, cy), 1, color, 2)
            #cv2.imshow("Circles", img)
            #cv2.waitKey(0)
     #   cv2.drawContours(img, cnt, -1, (0, 255, 0), 2)
    cv2.imshow("Circles", img)
    cv2.waitKey(0)
# loop over the contours individually
#for c in cnts:
    # if the contour is not sufficiently large, ignore it
#    if cv2.contourArea(c) < 100:
#        continue

def open_original_image(file_name):
    cv2.namedWindow("Input Image", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    cv2.resizeWindow("Input Image", 900, 600)
    image = cv2.imread(file_name)
    h = image.shape[0]
    w = image.shape[1]
    cv2.rectangle(image, (0, 0), (w-1, h-1), (255, 255, 255), 2)
    cv2.imshow("Input Image", image)
    cv2.waitKey(0)
    return image

if __name__ == '__main__':
    image = open_original_image(file_name)
    #find_contuors(image)

    contours = open_contours(contours_file)
    draw_contours(image, contours)
    draw_circles(image, contours)
    #turning_image(image)
    #canny_edge(image)
    #cv2.destroyAllWindows()