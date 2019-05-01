import os
from matplotlib.image import imread
from scipy.misc import imresize, imsave

#file path vars
parent_dir_1 = "../originalImages/Images"
parent_dir_2 = "dataset_updated_2"
sub_dir_1 = "training_set"
sub_dir_2 = "validation_set"
sub_sub_dir_1 = "drawings"
sub_sub_dir_2 = "engraving"
sub_sub_dir_3 = "iconography"
sub_sub_dir_4 = "painting"
sub_sub_dir_5 = "sculpture"
newdir = "../resizedImages/dogs128"

x_dim = 128
y_dim = x_dim
print(parent_dir_1)

filepaths = []
for dir, _, files in os.walk(parent_dir_1):
    for filename in files:
        relDir = os.path.relpath(dir, parent_dir_1)
        relFile = os.path.join(relDir, filename)
        filepaths.append(parent_dir_1 + "/" + relFile)

#filepaths[0] = filepaths[1] #dealing with DS_Store on Mac
for image, filepath in enumerate(filepaths):
    try:
        img = imread(filepath)
        img = imresize(img, (x_dim, y_dim))
        imsave(newdir + "/" + str(image) + ".jpg", img)
    except:
        pass

filepaths_new = []
for dir, _, files in os.walk(newdir):
    for filename in files:
        if not filename.endswith(".jpg"):
            continue
        relDir = os.path.relpath(dir, newdir)
        relFile = os.path.join(relDir, filename)
        filepaths_new.append(newdir + "/" + relFile)
