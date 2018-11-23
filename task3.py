import cv2
import numpy as np
import math

def template_matching(img_name, template, threshold=0):      
        # input image
        image = cv2.imread(img_name)

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # template image

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # apply laplacian of guassian on img
        blurred_img = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=0)
        laplacian_img = cv2.Laplacian(blurred_img, cv2.CV_32F)

        scales = np.linspace(0.05,0.6,20)


        matches = []
        # template scaled to different sizes
        for scale in scales:
            resized_template = cv2.resize(template_gray, (0, 0), fx=scale, fy=scale)
            laplacian_template = cv2.Laplacian(resized_template, cv2.CV_32F)

            if laplacian_img.shape[0] > laplacian_template.shape[0] and laplacian_img.shape[1] > laplacian_template.shape[1]:
                # if input size greater than template call match template using correlation coefficient
                result = cv2.matchTemplate(laplacian_img, laplacian_template, cv2.TM_CCORR)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
                # get the maximum value and index correspongin to the match 
                matches.append([maxVal, maxLoc])

        maxVal, maxLoc = max(matches, key=lambda elm: elm[0])
        print(img_name, maxVal)
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        (endX, endY) = (int((maxLoc[0] + template.shape[0])), int((maxLoc[1] + template.shape[1])))

        if maxVal > threshold:
            cv2.rectangle(image,(startX, startY), (endX, endY), 255, 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    template = cv2.imread("./task3/template.png")
    for i in range(1, 16):
        template_matching("./task3/pos_%d.jpg" % i, template)
    for j in range(1, 7):
        template_matching("./task3/neg_%d.jpg" % j, template, threshold=450000)
    for k in range(8, 11):
        template_matching("./task3/neg_%d.jpg" % k, template, threshold=450000)
    for i in range(1, 7):
        template_matching(("./task3_bonus/t3_%d.jpg" % i), template)
    template = cv2.imread("./task3_bonus/black_pointer.png")
    for i in range(1, 7):
        template_matching(("./task3_bonus/t2_%d.jpg" % i), template)
    template = cv2.imread("./task3_bonus/hand_pointer.png")
    for i in range(1, 7):
        template_matching(("./task3_bonus/t1_%d.jpg" % i), template)
    template = cv2.imread("./task3_bonus/hand_pointer.png")
    for j in range(1, 7):
        template_matching("./task3_bonus/neg_%d.jpg" % j, template, threshold=290000)
    for k in range(8, 11):
        template_matching("./task3_bonus/neg_%d.jpg" % k, template, threshold=290000)