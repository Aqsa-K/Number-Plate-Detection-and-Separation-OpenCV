import cv2                  # Importthe Opencv Library
import numpy as np          # Import NumPy, package for scientific computing with Python


img = cv2.imread('Car_Image_1.jpg')                     # Read the Image
cv2.namedWindow("Original Image",cv2.WINDOW_NORMAL)     # Create a Named window to display image
cv2.imshow("Original Image",img)                        # Display the Image

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.namedWindow("1 - Grayscale Conversion",cv2.WINDOW_NORMAL)
cv2.imshow("1 - Grayscale Conversion",img_gray)

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
cv2.namedWindow("2 - Noise Removal(Bilateral Filtering)",cv2.WINDOW_NORMAL)
cv2.imshow("2 - Noise Removal(Bilateral Filtering)",noise_removal)

# Histogram equalisation for better results
equal_histogram = cv2.equalizeHist(noise_removal)
cv2.namedWindow("3 - Histogram equalisation",cv2.WINDOW_NORMAL)
cv2.imshow("3 - Histogram equalisation",equal_histogram)

# Morphological opening with a rectangular structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))                                # create the kernel
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)     # Morphological opening using the kernal created
cv2.namedWindow("4 - Morphological opening",cv2.WINDOW_NORMAL)
cv2.imshow("4 - Morphological opening",morph_image)

# Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
cv2.namedWindow("5 - Image Subtraction", cv2.WINDOW_NORMAL)
cv2.imshow("5 - Image Subtraction", sub_morp_image)

# Thresholding the image
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
cv2.namedWindow("6 - Thresholding",cv2.WINDOW_NORMAL)
cv2.imshow("6 - Thresholding",thresh_image)

# Applying Canny Edge detection
canny_image = cv2.Canny(thresh_image,250,255)
cv2.namedWindow("7 - Canny Edge Detection",cv2.WINDOW_NORMAL)
cv2.imshow("7 - Canny Edge Detection",canny_image)

canny_image = cv2.convertScaleAbs(canny_image)

# Dilation - to strengthen the edges
kernel = np.ones((3,3), np.uint8)                               # Create the kernel for dilation
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)     # Carry out Dilation
cv2.namedWindow("8 - Dilation(closing)", cv2.WINDOW_NORMAL)
cv2.imshow("8 - Dilation(closing)", dilated_image)

# Finding Contours in the image based on edges
new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort the contours based on area ,so that the number plate will be in top 10 contours
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

NumberPlateCnt = None

# loop over the contours list
for c in contours:
     # approximate the contour
     peri = cv2.arcLength(c, True)
     approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
     # if our approximated contour has four points, then
     # we can assume that we have found our NumberPlate
     if len(approx) == 4:           # Select the contour with 4 corners
          NumberPlateCnt = approx   #assign to NumberPlateCnt when approximate contour found
          break                     # break the loop when Number Plate contour found/approximated

# Drawing the selected contour on the original image
final = cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

cv2.namedWindow("9 - Approximated Contour",cv2.WINDOW_NORMAL)
cv2.imshow("9 - Approximated Contour",final)


# SEPARATING OUT THE NUMBER PLATE FROM IMAGE:

# Masking the part other than the number plate
mask = np.zeros(img_gray.shape,np.uint8)                            # create an empty black image
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1,)       # Draw the contour of number plate on the black image - This is our mask
new_image = cv2.bitwise_and(img,img,mask=mask)                      # Take bitwise AND with the original image so we can just get the Number Plate from the original image
cv2.namedWindow("10 - Number Plate Separation",cv2.WINDOW_NORMAL)
cv2.imshow("10 - Number Plate Separation",new_image)



#HISTOGRAM EQUALIZATION FOR ENHANCING THE NUMBER PLATE FOR FURTHER PROCESSING:


y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))        # Converting the image to YCrCb model and splitting the 3 channels
y = cv2.equalizeHist(y)                                                 # Applying histogram equalisation
final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)    # Merging the 3 channels
cv2.namedWindow("11 - Enhanced Number Plate",cv2.WINDOW_NORMAL)
cv2.imshow("11 - Enhanced Number Plate",final_image)


cv2.waitKey()                                                           # Wait for a keystroke from the user before ending the program