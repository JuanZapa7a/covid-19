# USAGE
# python build_covid19_dataset.py --covid
# ../datasets/covid-chestxray-dataset-master --output
# ../datasets/dataset/covid

# import the necessary packages
import pandas as pd
from imutils import paths
import argparse
import random
import shutil
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-k", "--kaggle", default = "../datasets/chest_xray",
								help="path to base directory of Kaggle X-ray dataset")
ap.add_argument("-s", "--sample", type=int, default=180,
								help="# of samples to pull from Kaggle dataset")
ap.add_argument("-c", "--covid", default="../datasets/covid-chestxray-dataset-master",
	help="path to base directory for COVID-19 dataset")
ap.add_argument("-o", "--output", default="../datasets/dataset",
	help="path to directory where 'normal' and 'covid' images will be stored")
args = vars(ap.parse_args())

# grab all test,train and val image paths from the Kaggle X-ray dataset
# chest_xray
# ├── test
# │		 ├── NORMAL [234 entries exceeds filelimit, not opening dir]
# │		 └── PNEUMONIA [390 entries exceeds filelimit, not opening dir]
# ├── train
# │		 ├── NORMAL [1341 entries exceeds filelimit, not opening dir]
# │		 └── PNEUMONIA [3875 entries exceeds filelimit, not opening dir]
# └── val
# 		 ├── NORMAL [8 entries exceeds filelimit, not opening dir]
# 		 └── PNEUMONIA [8 entries exceeds filelimit, not opening dir]
#
# basepath is a concatenation with os.sep (/) as separator from args and args*
# imagepaths list of valid images below basepath (5856 images)
testbasePath = os.path.sep.join([args["kaggle"], "test", "NORMAL"])
trainbasePath = os.path.sep.join([args["kaggle"], "train", "NORMAL"])
valbasePath = os.path.sep.join([args["kaggle"], "val", "NORMAL"])
imagePaths = list(paths.list_images(testbasePath)) + list(paths.list_images(trainbasePath)) + list(paths.list_images(valbasePath))

# randomly sample the image paths
# I choose 25 first randomized images only NORMAL and PNEUMONIA and from
# test, train y val directories
random.seed(42)
random.shuffle(imagePaths)
imagePaths = imagePaths[:args["sample"]]

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the filename from imagePath and then construct the
	# path to the copied "normal" image file
	# copy the image from imagepath/filename to outpath/filename
	shutil.copy2(imagePath, os.path.sep.join([args["output"],"normal"]))

# construct the path to the metadata CSV file and load it
# covid-chestxray-dataset-master
# ├── docs
# │ 	├── covid-xray-umap.png
# │ 	└── share-image.png
# ├── images [884 entries exceeds filelimit, not opening dir]
# └── metadata.csv
#
# read metadata.csv
csvPath = os.path.sep.join([args["covid"], "metadata.csv"])
df = pd.read_csv(csvPath)

# loop over the rows of the COVID-19 data frame
for (i, row) in df.iterrows():
	# if (1) the current case is COVID-19 and (2) this is
	# a 'PA' view, then it is stored
	if row["finding"] == "COVID-19" and row["view"] == "PA":
		shutil.copy2(os.path.sep.join([args["covid"], "images", row["filename"]]),
								 os.path.sep.join([args["output"],"covid"]))
