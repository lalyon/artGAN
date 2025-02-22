import os
from matplotlib.image import imread, imsave # Keep imread and imsave from matplotlib for now, imsave might be replaced if needed.
# from scipy.misc import imresize, imsave # Removed scipy.misc imports
from PIL import Image # Import PIL Image for resizing and saving
from argparse import ArgumentParser
import sys
import numpy as np # Make sure numpy is imported

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument("-d", help="absolute path to directory with images")
parser.add_argument("-y", help="output y dimension")
parser.add_argument("-x", help="output x dimension")
parser.add_argument("-o", help="where to save cropped imgs")
args = parser.parse_args()

# Check the number of arguments
nArgs = len(sys.argv)
if nArgs != 5:
    print("len args incorrect. Expect 5, got ", nArgs)
else: 
    print("num of cli args: ", nArgs)
print(args)

# Convert dimensions to integers
y_dim = int(args.y)
x_dim = int(args.x)
outputDir = args.o
fullResDir = args.d

print("Cropping images in ", fullResDir)

# Collect file paths of images
filepaths = []
for dir, _, files in os.walk(fullResDir):
    for filename in files:
        relDir = os.path.relpath(dir, fullResDir)
        relFile = os.path.join(relDir, filename)
        # Temp fix for file path containing an extra "/./"
        relFile = relFile[2:]
        filepaths.append(fullResDir + "/" + relFile)

# Resize and save images
for image_index, filepath in enumerate(filepaths): # Use enumerate for index

    try: 
        img_np = imread(filepath) # Load image using matplotlib.image.imread, returns numpy array
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8)) # Convert numpy array to PIL Image. Rescale if needed.
        img_resized_pil = img_pil.resize((x_dim, y_dim), Image.LANCZOS) # Resize using PIL Image.LANCZOS for high-quality resizing

        # Save the PIL Image, using .jpg extension directly in filename
        output_filename = os.path.join(outputDir, f"{image_index}.jpg") # Use f-string for cleaner filename
        img_resized_pil.save(output_filename, 'JPEG') # Save as JPEG

        print(f"Resized and saved image {image_index+1}/{len(filepaths)}: {output_filename}") # Progress indicator

    except Exception as e: # Catch broad exceptions for robustness, print specific error
        print(f"Error processing image {filepath}: {e}")
        pass # Keep pass for now, consider more specific error handling

# Collect file paths of resized images
filepaths_new = []
for dir, _, files in os.walk(outputDir):
    for filename in files:
        if not filename.endswith(".jpg"):
            continue
        relDir = os.path.relpath(dir, outputDir)
        relFile = os.path.join(relDir, filename)
        filepaths_new.append(outputDir + "/" + relFile)

print(f"\nSuccessfully resized and saved images to: {outputDir}")
print(f"Total resized images: {len(filepaths_new)}")