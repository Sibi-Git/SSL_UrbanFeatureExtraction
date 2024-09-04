import os

#c1 = 0
c = 1

for file_name in os.listdir('/home/ds6812/cvproj2/data/test/1/'):
    os.rename('/home/ds6812/cvproj2/data/test/1/' + file_name, '/home/ds6812/cvproj2/data/test/1/' + str(c) + '.tif')
    #os.rename('/scratch/ds6812/dota_reduced/validation/labels/' + str(c1) + '.yml', '/scratch/ds6812/dota_reduced/validation/labels/' + str(c2) + '.yml')
    #c1+=1
    c+=1

