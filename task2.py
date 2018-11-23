import cv2
import numpy as np
import math


def scaledown_half(img):
    # subsample every other pixel in the image
    resized_img = []
    for i in range(0, img.shape[0], 2):
        row = []
        for j in range(0, img.shape[1], 2):
            row.append(img[i, j])
        resized_img.append(row)
    return np.array(resized_img)

def scaleup_twice(img):
    # scales the image up by twice its dimensions by redundancy
    enlarged_img = []
    for i in range(img.shape[0]):
        if i+1 < img.shape[0]:
            row = []
            for j in range(img.shape[1]):
                if j+1 < img.shape[1]:
                    row.append(img[i, j])
                    row.append(img[i, j+1])
            enlarged_img.append(row)
            enlarged_img.append(row)
    return np.array(enlarged_img)

def flatten_arr(arr):
    result = []
    for i in range(len(arr)):
        for j in range(len(arr[0])):
            result.append(arr[i][j])
    return result

# convolution operation
def conv(img, kernel, kernel_size):
    result = []
    kernel = kernel[::-1, ::-1]
    for i in range(img.shape[0]):
        if i <= img.shape[0] - kernel_size:
            row = []
            for j in range(img.shape[1]):
                if j <= img.shape[1] - kernel_size:
                    arr = flatten_arr( kernel * img[i:i+kernel_size, j:j+kernel_size])
                    row.append(sum(arr))
            result.append(row)
    return np.array(result)


def get_guassian(x, y, sigma):
    return (1 / (sigma**2 * (2 * math.pi))) * (math.exp(- (x**2 + y**2) / (2 * sigma**2)))


def guassian_blur(img, ksize, sigma):
    # blurs the input image using guassian distribution
    guassian = []
    a = []
    for j in range(2, -3, -1):
        temp = []
        for i in range(-2, 3):
            b = get_guassian(i, j, sigma)
            a.append(b)
            temp.append(b)
        guassian.append(temp)
    guassian = np.array(guassian) / sum(a)
    # now convolve guassian with img
    return conv(img, guassian, ksize[0])

def get_keypoint_indices(key_img):
    mask = (key_img == 255)
    points_arr = []
    [points_arr.append([i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1]) if mask[i][j] == True]
    return points_arr

def overlay_keypoints(img, key_arr):
    # gets keypoint indices and paints those in red color on color image
    for key_img in key_arr:
        for [x, y] in get_keypoint_indices(key_img):
            img[x, y] = [0, 0, 255]

    return img

def generate_octaves(img):
    # generates 4 octaves
    octaves = []
    octaves.append(img)
    current_img = img
    for i in range(3):
        resized_img = scaledown_half(current_img)
        octaves.append(resized_img)
        current_img = resized_img
        # cv2.namedWindow("", cv2.WINDOW_AUTOSIZE)
        # cv2.imshow("", resized_img)
        # cv2.waitKey(0)
    return octaves

def generate_guassians(octaves, sigmas):
    # create gaussian blurs for images in each octave
    octave_guassians = []
    for octave, sigma_list in zip(octaves, sigmas):
        guassian_blurs = []
        for sigma in sigma_list:
            blur = guassian_blur(octave, (5, 5), sigma)
            guassian_blurs.append(blur)
        octave_guassians.append(guassian_blurs)
    return octave_guassians

def generate_dogs(octave_guassians):
    # takes difference between guassian blurred images
    octave_dogs = []
    for guassian_group in octave_guassians:
        dog_list = []
        i = 0
        while i < len(guassian_group) - 1:
            dog = guassian_group[i] - guassian_group[i - 1]
            dog_flatten = flatten_arr(dog)
            dog = (dog - min(dog_flatten)) / (max(dog_flatten) - min(dog_flatten))
            i += 1
            dog_list.append(dog)
        octave_dogs.append(dog_list)
    return octave_dogs


def generate_keypoints(m1, m2, m3):
    # generates keypoints on the image by comparing maxima between DoGs
    result = []
    for i in range(m1.shape[0]):
        if i <= m1.shape[0] - 3:
            row = []
            for j in range(m1.shape[1]):
                if j <= m1.shape[1] - 3:
                    m1_flatten = flatten_arr(m1[i:i+3, j:j+3])
                    m2_flatten = flatten_arr(m2[i:i+3, j:j+3])
                    m3_flatten = flatten_arr(m3[i:i+3, j:j+3])

                    m2_max = max(m2_flatten)
                    m3_max = max(m3_flatten)
                    m1_max = max(m1_flatten)
                    if m1[i+1, j+1] != m1_max or m1[i+1, j+1] < m2_max or m1[i+1, j+1] < m3_max:
                        row.append(0)
                    else:
                        row.append(255)
            result.append(row)
    return result

if __name__ == "__main__":
        
    # read source image
    img = cv2.imread("./task2.jpg", 0)

    print("Generating 4 octaves ....")
    octaves = generate_octaves(img)

    sigmas = [[2**-0.5, 1, 2**0.5, 2, 2**1.5],
            [2**0.5, 2, 2**1.5, 4, 2**2.5],
            [2**1.5, 4, 2**2.5, 8, 2**3.5],
            [2**2.5, 8, 2**3.5, 16, 2**4.5]
            ]
    print("Generating 5 Guassian blurs each for 4 octaves .... (takes approx. 1 minute)")
    octave_guassians = generate_guassians(octaves, sigmas)

    print("Generating (5-1)=4 DoGs each for 4 octaves .....")
    octave_dogs = generate_dogs(octave_guassians)

    print("Generating 2 keypoint images each for 4 octaves .....")
    keypoint_groups = []
    for dog_group in octave_dogs:
        temp1 = generate_keypoints(dog_group[0], dog_group[1], dog_group[2])
        temp2 = generate_keypoints(dog_group[0], dog_group[2], dog_group[3])

        keypoint_groups.append([np.array(temp1), np.array(temp2)])

    for kg in keypoint_groups:
        for kp in kg:
            cv2.namedWindow("", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("", np.array(kp, dtype=np.uint8))
            cv2.waitKey(0)
    

    merged_keypoints = []
    merged_keypoints.append(keypoint_groups[0][0])
    merged_keypoints.append(keypoint_groups[0][1])

    for key_group in keypoint_groups[1:]:
        for key_img in key_group:
            merged_keypoints.append(scaleup_twice(key_img))

    color_img = cv2.imread("./task2.jpg")

    print("Overlaying keypoints on the color image .....")
    key_img = overlay_keypoints(color_img, merged_keypoints)

    cv2.namedWindow("", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("", key_img)
    cv2.waitKey(0)
