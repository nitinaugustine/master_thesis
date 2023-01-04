import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def gaus2d(x, y, sx, peak):
    """
    function that defines the super gaussian
    in 2d.
    sx -- defines the spread of the gaussian
    peak -- defines the peak contrast intensity
    the power of power_x and power_y defines the level of 
    flat headedness.

    for more information on super_gaussian function
    check https://www.hellenicaworld.com/Science/Mathematics/en/GaussianFunction.html
    """
    sy=sx
    power_x = x**2. / (2. * sx**2.)
    power_y = y**2. / (2. * sy**2.)
    P=2.7
    return peak *(1. / (2. * np.pi * sx * sy) * np.exp(-(power_x + power_y)**P))

def gaussian(size,int_drop):

    """
    function that defines the mesh for creating
    the 2d gaussain
    the space is created in accordance with the
    size of the blemish
    sx -- defines the standard deviation that controls the
          spread of the gaussian
    """

    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    spread = 0.5
    intensity_drop = int_drop
    print('intensity_drop',intensity_drop)
    z = gaus2d(x, y, spread, intensity_drop)
    blemish = 1-z

    #plt.imshow(blemish,cmap='gray')
    #plt.show()
    
    return blemish

def diff_corrector(value1,value2):
    diff = value1 - value2
    return value1 + diff

def blemish_gen(image,size,loc,img_size,int_drop):
    
    r,c = loc
    r_min = round(r - size*0.5)
    r_max = round(r + size*0.5)
    c_min = round(c - size*0.5)
    c_max = round(c + size*0.5)

    fn_r_start = 0
    fn_r_end = size
    fn_c_start = 0
    fn_c_end = size

    r_diff = r_max - r_min
    fn_r_diff = fn_r_end - fn_r_start

    c_diff = c_max - c_min
    fn_c_diff = fn_c_end - fn_c_start


    if r_diff!= fn_r_diff:
        diff = fn_r_diff - r_diff
        r_max = r_max + diff
        roi = image[r_min:r_max,c_min:c_max]
        print("shape diff1")

    if c_diff!= fn_c_diff:
        diff = fn_c_diff - c_diff
        c_max = c_max + diff
        roi = image[r_min:r_max,c_min:c_max]
        print("shape diff2")

    print("-----INITIALIZE----)")

    print("ROI dim:",r_min,r_max,c_min,c_max)
    print("ROI dim:",fn_r_start,fn_r_end,fn_c_start,fn_c_end)

    print("---------)")
 
    
    
    fn = gaussian(size,int_drop)
    fn = np.round_(fn,3)
    #plt.imshow(fn,cmap='gray')
    #plt.show()

    if c_min<0:
        fn_c_start = (-1*c_min)
        c_min = 0
    elif c_max>img_size[1]:
        fn_c_end = size-(c_max-img_size[1])
        c_max = img_size[1]+1

    if r_min<0:
        r_min =  0
        fn_r_start = (-1*r_min)
    elif r_max>img_size[0]:
        fn_r_end = size-(r_max-img_size[0])
        r_max = img_size[0]+1

    roi = image[r_min:r_max,c_min:c_max]
    #print('ROI shape',roi.shape)

    roi_fn = fn[fn_r_start:fn_r_end,fn_c_start:fn_c_end]
    #print('fn shape',roi_fn.shape)

    print("-----modified----)")

    print("ROI dim:",r_min,r_max,c_min,c_max)
    print("ROI dim:",fn_r_start,fn_r_end,fn_c_start,fn_c_end)

    print("---------)")
 

    #if roi.shape[0]!= fn.shape[0]:
        #diff = fn.shape[0]-roi.shape[0]
        #r_max = r_max + diff
        #roi = image[r_min:r_max,c_min:c_max]
        #print("shape diff")

    #if roi.shape[1]!= fn.shape[1]:
        #diff = fn.shape[1]-roi.shape[1]
        #c_max = c_max + diff
        #roi = image[r_min:r_max,c_min:c_max]
        #[print("shape diff")]


    print('ROI shape',roi.shape)
    print('fn shape',roi_fn.shape)



    r_roi = roi[:,:,0]
    r_channel = r_roi * fn[fn_r_start:fn_r_end,fn_c_start:fn_c_end]

    g_roi = roi[:,:,1]
    g_channel = g_roi * fn[fn_r_start:fn_r_end,fn_c_start:fn_c_end]

    b_roi = roi[:,:,2]
    b_channel = b_roi * fn[fn_r_start:fn_r_end,fn_c_start:fn_c_end]

    image[r_min:r_max,c_min:c_max,0] = r_channel
    image[r_min:r_max,c_min:c_max,1] = g_channel
    image[r_min:r_max,c_min:c_max,2] = b_channel

    #cv2.imwrite('H:/THI- MAPE/Master Thesis/resources/blemished/expjz2.png', image)
    
    return image

