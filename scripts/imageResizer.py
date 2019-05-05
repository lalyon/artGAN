import os
from matplotlib.image import imread
from scipy.misc import imresize, imsave
from argparse import ArgumentParser
import sys


# parser = ArgumentParser()
# parser.add_argument("-d", help="absolute path to directory with images")
# parser.add_argument("-y", help="output y dimension")
# parser.add_argument("-x", help="output x dimension")
# parser.add_argument("-o", help="where to save cropped imgs")
# args = parser.parse_args()

# nArgs = len(sys.argv)
# if nArgs != 5:
#     print("len args incorrect. Expect 5, got ", nArgs)
# else: print("num of cli args: ", nArgs)
# print(args)
# y_dim = args.y
# x_dim = args.x
y_dim = 256
x_dim = 256
outputDir="/home/luke/Documents/artGAN/portraits256"
fullResDir="/home/luke/Documents/artGAN/portraitsFullRes/portrait"
# fullResDir = args.d
# outputDir = args.o
print("Cropping images in ", fullResDir)

filepaths = []
for dir, _, files in os.walk(fullResDir):
    for filename in files:
        relDir = os.path.relpath(dir, fullResDir)
        relFile = os.path.join(relDir, filename)
        #temp fix for file path containing an extra "/./"
        relFile = relFile[2:]
        filepaths.append(fullResDir + "/" + relFile)

#filepaths[0] = filepaths[1] #dealing with DS_Store on Mac
for image, filepath in enumerate(filepaths):

    try:
        img = imread(filepath)
        img = imresize(img, (x_dim, y_dim))
        imsave(outputDir + "/" + str(image) + ".jpg", img)
    except: pass

filepaths_new = []
for dir, _, files in os.walk(outputDir):
    for filename in files:
        if not filename.endswith(".jpg"):
            continue
        relDir = os.path.relpath(dir, outputDir)
        relFile = os.path.join(relDir, filename)
        filepaths_new.append(outputDir + "/" + relFile)
