import os
import shutil 

unlabeledImagesDirectory = '/Users/deeptisaravanan/Desktop/Fall_2022/Computer_Vision/Project/unlabeled/images/'
filenames = os.listdir(unlabeledImagesDirectory)
file_tracker = {}
index = 0
for i in range(len(filenames)):
    newFilename = str(index) + '.tif'
    file_tracker[index] = filenames[i]
    shutil.move(unlabeledImagesDirectory + filenames[i], unlabeledImagesDirectory + newFilename)
    index = index + 1

print(file_tracker.keys())