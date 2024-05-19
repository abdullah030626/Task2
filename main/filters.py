from PIL import Image , ImageOps , ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def get_erosion(image,kernal_size=5):
    image_gray=cv.cvtColor(np.array(image),cv.COLOR_RGB2GRAY)
    kernel = np.ones((kernal_size, kernal_size), np.uint8) 
    img_erosion = cv.erode(image_gray, kernel, iterations=1)
    return img_erosion

def get_dilation(image,kernel_size):
    image_gray=cv.cvtColor(np.array(image),cv.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    img_dilation = cv.dilate(image_gray, kernel, iterations=1)
    return img_dilation

def get_open(image,kernel_size):
    image_gray=cv.cvtColor(np.array(image),cv.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    open_image = cv.morphologyEx(image_gray, cv.MORPH_OPEN, kernel)
    return open_image

def get_close(image,kernel_size):
    image_gray=cv.cvtColor(np.array(image),cv.COLOR_RGB2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    close_image = cv.morphologyEx(image_gray, cv.MORPH_CLOSE, kernel)
    return close_image  

def get_hough_transform(image):
    image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)  # Convert to numpy array and grayscale
    image_gray = cv.medianBlur(image_gray,5)

    cimg = cv.cvtColor(image_gray,cv.COLOR_GRAY2RGB)
    circles = cv.HoughCircles(image_gray,cv.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=5,maxRadius=100)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
         # draw the outer circle
         cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
         # draw the center of the circle
         cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    return cimg

def get_seg_split_and_merge(image):
    # # Convert numpy array to PIL Image
    # pil_image = Image.fromarray(image)
    # Convert image to grayscale
    gray = image.convert('L')

    width, height = image.size
    segmented_image = np.zeros((height, width), dtype=np.uint8)

    # Define recursive function for region split and merge
    def split_merge(x, y, w, h):
        # Split region into four quadrants
        if w * h > 100:
            half_w = w // 2
            half_h = h // 2
            split_merge(x, y, half_w, half_h)
            split_merge(x + half_w, y, half_w, half_h)
            split_merge(x, y + half_h, half_w, half_h)
            split_merge(x + half_w, y + half_h, half_w, half_h)
        else:
            # Merge region if homogeneous
            region_mean = np.mean(np.array(gray.crop((x, y, x+w, y+h))))
            segmented_image[y:y+h, x:x+w] = 255 if region_mean > 128 else 0

    # Start region split and merge
    split_merge(0, 0, width, height)

    return segmented_image

    
def get_seg_threshold(image, value):
    image_gray=cv.cvtColor(np.array(image),cv.COLOR_RGB2GRAY)
    _, threshold_image = cv.threshold(image_gray, value, 255, cv.THRESH_BINARY)
    return cv.cvtColor(threshold_image, cv.COLOR_GRAY2RGB)

# Low Pass Filter (LPF)
def apply_lpf(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
    return cv.filter2D(src=image, ddepth=-1,kernel= kernel)

# High Pass Filter (HPF)
def apply_hpf(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv.filter2D(src=image, ddepth=-1, kernel=kernel)

# Mean Filter
def apply_mean(image, kernel_size):
    return cv.blur(image, (kernel_size, kernel_size))

# Median Filter
def apply_median(image, kernel_size):
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1  # Make it odd by adding 1

    # Apply median blur
    return cv.medianBlur(image, kernel_size)
# Roberts Edge Detection
def apply_roberts(image):
    image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    kernel_x = np.array([[2, 0], [0, -2]])
    kernel_y = np.array([[0, 2], [-2, 0]])
    image_x = cv.filter2D(image_gray, -1, kernel_x)
    image_y = cv.filter2D(image_gray, -1, kernel_y)
    return cv.addWeighted(image_x, 0.5, image_y, 0.5, 0)

# Prewitt Edge Detection
def apply_prewitt(image):
    image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    kernel_x = np.array([[-2, 0, 2], [-2, 0, 2], [-2, 0, 2]])
    kernel_y = np.array([[-2, -2, -2], [0, 0, 0], [2, 2, 2]])
    image_x = cv.filter2D(image_gray, -1, kernel_x)
    image_y = cv.filter2D(image_gray, -1, kernel_y)
    return cv.addWeighted(image_x, 0.5, image_y, 0.5, 0)

# Sobel Edge Detection
def apply_sobel(image):
    image_gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
    image_x = cv.Sobel(image_gray, cv.CV_64F, 1, 0, ksize=3)
    image_y = cv.Sobel(image_gray, cv.CV_64F, 0, 1, ksize=3)
    sobel_image = np.sqrt(image_x ** 2 + image_y ** 2)
        # Normalize the gradient magnitude image
    sobel_image = cv.normalize(sobel_image, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    sobel_image= cv.addWeighted(image_x,0.5,image_y,0.5,0)

    return sobel_image


