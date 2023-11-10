#filename = 'H:/THI- MAPE/Master Thesis/resources/corrected labels/005000.txt'

path1 = 'H:/THI- MAPE/Master Thesis/resources/corrected labels/'
path2 = 'H:/THI- MAPE/Master Thesis/resources/test_cor/'
path3 = 'H:/THI- MAPE/Master Thesis/resources/training/image_2/'

list1 = []
list2 = []

for file_ in enumerate(os.listdir(path1)):
    filename1 = path1 + file_[1]
    filename2 = path2 + file_[1]

    image_name = file_[1].split(".")[0] + '.png'

    img = cv2.imread(path3 + image_name)
    size = img.shape

    

    with open(filename1) as file:
        list2 = []
        
        for line in file:
            list1 = []
            
            #print(line.rstrip()) 
            values = line.split() #splits the values of every line in the text file to individual values
            dw = 1. / size[1]
            dh = 1. / size[0]

            if values[0] =='Car':
                label = 0
            elif values[0] =='Van':
                label = 1
            elif values[0] =='Truck':
                label = 2
            elif values[0] =='Pedestrian':
                label = 3
            elif values[0] =='Person':
                label = 4
            elif values[0] =='Cyclist':
                label = 5
            elif values[0] =='Tram':
                label = 6
            elif values[0] =='Misc':
                label = 7

    
            x = (float(values[1]) + float(values[3])) / 2.0
            y = (float(values[2]) + float(values[4])) / 2.0
            w = float(values[3]) - float(values[1])
            h = float(values[4]) - float(values[2])
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            list1.extend([label,x,y,w,h])
            

            #print('\nlist 1 -->',list1)
            list2.append(list1)
            #print('list2-->\n', list2)
            #print('\n')
            
        with open(filename2, 'w') as fp:
            for value in list2:
                fp.write(f'{value[0]}' + ' ' + f'{value[1]}' + ' ' + f'{value[2]}'+ ' ' + f'{value[3]}' + ' ' + f'{value[4]}')
                fp.write('\n')

    print('Completed:',file_[1])
    print('\n')
