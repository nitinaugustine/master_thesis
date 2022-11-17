# BLEMISH CODE

bk = np.ones((374, 1238,3),dtype=np.uint8)*255
blem = cv2.circle(bk,(619,187),25,(0,0,0),-1)
blem = cv2.circle(bk,(900,187),25,(0,0,0),-1)
blem = cv2.circle(bk,(338,187),25,(0,0,0),-1)
blem = cv2.circle(bk,(619,287),25,(0,0,0),-1)
blem = cv2.circle(bk,(619,87),25,(0,0,0),-1)
plt.imshow(blem)
plt.show()

# read the image
image_bgr = blem.copy()
# get the image dimensions (height, width and channels)
h, w, c = bg.shape
# append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
# create a mask where white pixels ([255, 255, 255]) are True
white = np.all(image_bgr == [255, 255, 255], axis=-1)
# change the values of Alpha to 0 for all the white pixels
image_bgra[white, -1] = 0
# save the image
bg = cv2.GaussianBlur(image_bgra,ksize=(21,21),sigmaX=20,sigmaY=20)
cv2.imwrite('H:/THI- MAPE/Master Thesis/resources/whitebk.png', bg)

background = cv2.imread('H:/THI- MAPE/Master Thesis/resources/training/image_2/000017.png')
overlay = cv2.imread('H:/THI- MAPE/Master Thesis/resources/whitebk.png', cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel

# separate the alpha channel from the color channels
alpha_channel = overlay[:, :, 3] / 255 # convert from 0-255 to 0.0-1.0
overlay_colors = overlay[:, :, :3]

# To take advantage of the speed of numpy and apply transformations to the entire image with a single operation
# the arrays need to be the same shape. However, the shapes currently looks like this:
#    - overlay_colors shape:(width, height, 3)  3 color values for each pixel, (red, green, blue)
#    - alpha_channel  shape:(width, height, 1)  1 single alpha value for each pixel
# We will construct an alpha_mask that has the same shape as the overlay_colors by duplicate the alpha channel
# for each color so there is a 1:1 alpha channel for each color channel
alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

# The background image is larger than the overlay so we'll take a subsection of the background that matches the
# dimensions of the overlay.
# NOTE: For simplicity, the overlay is applied to the top-left corner of the background(0,0). An x and y offset
# could be used to place the overlay at any position on the background.
h, w = overlay.shape[:2]
background_subsection = background[0:h, 0:w]

# combine the background with the overlay image weighted by alpha
composite = background_subsection * (1 - alpha_mask) + overlay_colors * alpha_mask

# overwrite the section of the background image that has been updated
background[0:h, 0:w] = composite

cv2.imwrite('H:/THI- MAPE/Master Thesis/resources/combined.png', background)