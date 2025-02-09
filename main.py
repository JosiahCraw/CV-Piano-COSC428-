import numpy as np
import cv2


def nothing(x):
    pass


def line_vertical(x0, y0, x1, y1) -> bool:
    vert_hor_ratio = abs(x0-x1)/abs(y0-y1)
    if vert_hor_ratio <= 0.3:
        return True
    elif vert_hor_ratio >= 2:
        return False
    return None


def houghNormal(img_original):
    gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 15)

    # Create a new copy of the original image for drawing on later.
    img = img_original.copy()
    # Use the Canny Edge Detector to find some edges.
    edges = cv2.Canny(gray, 1000, 1200)
    # Attempt to detect straight lines in the edge detected image.
    lines = cv2.HoughLines(edges, 1, np.pi/180, 53)

    output_lines = list()
    vertical_lines = list()
    horizontal_lines = list()

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                vertical = line_vertical(x0, y0, x1, y1)
                if vertical == True:
                    # output_lines.append([(x1, y1), (x2, y2)])
                    vertical_lines.append([(x1, y1), (x2, y2)])
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 130)

    # For each line that was detected, draw it on the img.
    if lines is not None:
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                vertical = line_vertical(x0, y0, x1, y1)
                if vertical == False:
                    # output_lines.append([(x1, y1), (x2, y2)])
                    horizontal_lines.append([(x1, y1), (x2, y2)])
                    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    return img, vertical_lines, horizontal_lines


def erosion(img):
    gaussian_kernel = 30 // 2 * 2 + 1
    morph_iterations = 6
    img_blur = cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
    # Use adaptive thresholding to "binarize" the image.
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 1)

    # Perform some morphological operations to help distinguish some of the features in the image.
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=morph_iterations)

    return erosion


def dilation(img):
    gaussian_kernel = 10 // 2 * 2 + 1
    morph_iterations = 1

    # Gaussian blur to reduce noise in the image.
    img_blur = cv2.GaussianBlur(img, (gaussian_kernel, gaussian_kernel), 0)
    # Use adaptive thresholding to "binarize" the image.
    thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 1)

    # Perform some morphological operations to help distinguish some of the features in the image.
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=morph_iterations)

    return dilation

def get_slope_offset(line):
    x0 = line[0][0]
    x1 = line[0][1]
    y0 = line[1][0]
    y1 = line[1][0]
    m = (y1-y0)/(x1-x0)
    if m == 0:
        m = 0.001
    c = y0

    return m, c

def get_intersection(m0, c0, m1, c1):
    print(m0)
    print(m1)
    print(m0-m1)
    if m1 == m0:
        m0 += 0.001
    x = (c1 - c0) / (m0 - m1)
    y = m0 * x + c0

    return (int(round(x)), int(round(y)))

def get_hough_intersections(vertical_lines, horizontal_lines):
    intersections = list()
    for vert_line in vertical_lines:
        for hori_line in horizontal_lines:
            hm, hc = get_slope_offset(hori_line)
            vm, vc = get_slope_offset(vert_line)

            intersections.append(get_intersection(hm, hc, vm, vc))

    return intersections

def get_rect(x, y, vertical_lines, horizontal_lines, img):
    img_dim = img.shape[:2]
    curr_below = None
    curr_above = None
    curr_left = None
    curr_right = None
    d_below = -10000
    d_above = img_dim[0]
    d_left = -10000
    d_right = img_dim[1]
    # print(len(horizontal_lines))
    # print(len(vertical_lines))
    print("Finding Rectangle")
    print('x: {}, y: {}'.format(x,y))
    print('Horizontal')
    for line in horizontal_lines:
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[1][0]
        y1 = line[1][1]
        if (y0 - y1) == 0:
            line_y = y0 - y1
        else:
            line_y = (y0-y1)/(x0-x1)*x+y0
        dy = line_y - y
        print('Line x0: {}, y0: {}, x1: {}, y1: {}'.format(x0, y0, x1, y1))
        print('dy: {}'.format(dy))
        if line_y < y:
            if line_y > d_below:
                d_below = line_y
                curr_below = line
        if line_y > y:
            if line_y < d_above:
                d_above = line_y
                curr_above = line
    print('Vertical')
    for line in vertical_lines:
        x0 = line[0][0]
        y0 = line[0][1]
        x1 = line[1][0]
        y1 = line[1][1]
        if (x0 - x1) == 0:
            line_x = x0
        else:
            line_x = (y - y0)/((y0-y1)/(x0-x1))
        dx = line_x - x
        print('Line x0: {}, y0: {}, x1: {}, y1: {}'.format(x0, y0, x1, y1))
        print('dx: {}'.format(dx))
        if line_x > x:
            if line_x < d_right:
                d_right = line_x
                curr_right = line
        if line_x < x:
            if line_x > d_left:
                d_left = line_x
                curr_left = line
    print('Chosen Line: L: {}, R: {}, A: {}, B: {}'.format(curr_left, curr_right, curr_above, curr_below))
    return (curr_left, curr_right, curr_above, curr_below)



def main():
    cap = cv2.VideoCapture('./output.avi')  # Open the webcam device.
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('hsv.avi', fourcc, 20.0, (640, 480))
    high_H = high_S = high_V = 255
    # Load two initial images from the webcam to begin.
    ret, img0 = cap.read()
    ret, img1 = cap.read()

    cv2.imshow('Paper Piano', img0)
    hough_img, vertical_lines, horizontal_lines = houghNormal(img0)

    cv2.createTrackbar('High H', 'Paper Piano' , 0, 255, nothing)
    cv2.createTrackbar('High S', 'Paper Piano' , 0, 255, nothing)
    cv2.createTrackbar('High V', 'Paper Piano' , 0, 255, nothing)   
    cv2.namedWindow('Paper Piano')
    prev_cx = None
    prev_cy = None
    
    while True:
        high_H = cv2.getTrackbarPos('High H', 'Paper Piano')
        high_S = cv2.getTrackbarPos('High S', 'Paper Piano')
        high_V = cv2.getTrackbarPos('High V', 'Paper Piano')
        # Calculate the differences of the two images.
        diffThreshold = cv2.getTrackbarPos('Differencing', 'Paper Piano')
        
        hsv_img = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_img, (0, 128, 80), (255, 255, 255))
        hsv_img = cv2.bitwise_and(img0, img0, mask=mask)


        diff = cv2.subtract(cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY),
                                            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        ret, diff = cv2.threshold(diff, 120, 255, cv2.THRESH_BINARY)
        # Move the data in img0 to img1. Uncomment this line for differencing from the first frame.
        img1 = img0
        ret, img0 = cap.read()  # Grab a new frame from the camera for img0.
        frame_cpy = img0
        # Use the moments of the difference image to draw the centroid of the difference image.
        moments = cv2.moments(diff)
        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            cv2.circle(diff, (cX, cY), 8, (255, 255, 255), -1)

        diff_thresh = erosion(diff)
        diff_thresh = dilation(diff_thresh)
        diff_thresh = cv2.bitwise_not(diff_thresh)
        diff_resize = cv2.resize(diff_thresh, (img1.shape[1], img1.shape[0]))
        diff_colour = cv2.cvtColor(diff_resize, cv2.COLOR_GRAY2BGR)
        moments = cv2.moments(diff)

        for line in vertical_lines:
            cv2.line(frame_cpy,line[0],line[1],(0,0,255),2)
        for line in horizontal_lines:
            cv2.line(frame_cpy,line[0],line[1],(0,0,255),2)

        if moments["m00"] != 0:  # Check for divide by zero errors.
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            if prev_cx != None and prev_cy != None:
                # cv2.circle(img0, (cX, cY), 8, (255, 0, 0), -1)
                dx = prev_cx - cX
                dy = prev_cy - cY
                print('dx: {}, dy: {}'.format(dx, dy))
                if dy < -1:
                    lines = get_rect(cX, cY, vertical_lines, horizontal_lines, diff_colour)
                    for line in lines:
                        if line != None:
                            print('{}, {}'.format(cX, cY))
                            cv2.line(frame_cpy, (line[0][0], line[0][1]), (line[1][0], line[1][1]), (255,0,0), 2)
                    cv2.circle(frame_cpy, (cX, cY), 8, (0, 0, 255), -1)
            prev_cx = cX
            prev_cy = cY
        diff_colour = diff_colour
        
        show_img = np.concatenate((frame_cpy, hough_img), axis=1)
        out.write(hsv_img) 
        # exit(0)
        cv2.imshow('Paper Piano', show_img)  # Display the difference to the screen.

        # Close the script when q is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # houghNormal()
    # segments()
