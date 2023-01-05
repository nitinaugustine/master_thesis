from blemish_generator import blemish_gen

"""
generate blemishes based on the labels and images
"""

"""
path1 : put the labels (with actual x y coordinates of objects)
path2 : the location where output images get stored
path3 : the location where images to be blemished is given
blemish_size : define the diameter of blemish in pixel
intensity_drop : how much opaque must the blemish be (preferable between 0.1 to 1)

Option1: generate blemish of fixed radii on all objects
Option2: generate blemish based on a defined ratio times the size of the object
"""
path3 = 'H:/THI- MAPE/Master Thesis/resources/exp1_images/'
path1 = 'H:/THI- MAPE/Master Thesis/resources/exp1_label/'
path2 = 'H:/THI- MAPE/Master Thesis/resources/exp1_op/'
Option1 = False
a_ratio = 0.5
blemish_size = 100
intensity_drop = 0.6


print('------------------------')
print('BLEMISH CHARACTERISTICS')
print('Blemish Diameter: ',blemish_size)
print('INTENSITY DROP (0.1-1): ',intensity_drop)
print('-----------------------')

print("\nSimulation started....please wait")


for file_ in enumerate(os.listdir(path1)):


    filename1 = path1 + file_[1]
    filename2 = path2 + file_[1]

    image_name = file_[1].split(".")[0] + '.png'

    img = cv2.imread(path3 + image_name)
    img_size = img.shape
    print("Image : ", image_name)

    with open(filename1) as file:
        for line in file:
            values = line.split() #splits the values of every line in the text file to individual values
            x = round((float(values[1]) + float(values[3])) / 2.0)
            y = round((float(values[2]) + float(values[4])) / 2.0)
            print('Blemish generated at ' + f'{x,y}')
            if Option1 is True:
                blem = blemish_gen(img,blemish_size,(y,x),img_size,intensity_drop)
            else:
                blemish_size = area_ratio(values,a_ratio)
                blem = blemish_gen(img,blemish_size,(y,x),img_size,intensity_drop)

        cv2.imwrite(path2+image_name,blem)
    print('Completed:',file_[1])
    print('\n')

print('Simulation completed successfully. Images are saved at' + f'{path2}' )