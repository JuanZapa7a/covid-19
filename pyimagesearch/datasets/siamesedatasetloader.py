# import the necessary packages
import numpy as np
import cv2
import os


class SiameseDatasetLoader:
    def __init__(self, preprocessors=None):
        # store the image preprocessor
        self.preprocessors = preprocessors

        # if the preprocessors are None, initialize them as an
        # empty list
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose=-1):
        # initialize the list of features and labels
        data = []
        labels = []
        imagePathsSorted = sorted(imagePaths)
        # loop over the input images
        # Genuine data: Choose the 10 images for each group (40) 4000
        for i in range(0, len(imagePathsSorted), 10):
            # load the image and extract the class label assuming
            # that our path has the following format:
            # /path/to/dataset/{class}/{image}.jpg
            # print("[INFO] loading {}".format(imagePath))
            l = imagePathsSorted[i:i + 10]
            m = l
            for j in range(10):
                for k in range(10):
                    images = []
                    image = cv2.imread(m[k])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # check to see if our preprocessors are not None
                    if self.preprocessors is not None:
                        # loop over the preprocessors and apply each to
                        # the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                    images.append(image)
                    image = cv2.imread(l[k])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # check to see if our preprocessors are not None
                    if self.preprocessors is not None:
                        # loop over the preprocessors and apply each to
                        # the image
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                    images.append(image)
                    # treat our processed image as a "feature vector"
                    # by updating the data list followed by the labels
                    data.append(images)
                    labels.append([1])
                # rotating the images in the list
                l = l[1:] + l[:1]

            # # show an update every `verbose` images
            # if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            #     print("[INFO] processed {}/{}".format(i + 1,
            #                                           len(imagePaths)))
        # Ungenuine  data: choose 3 random index
        count = 0
        while count != 4000:
            ind1 = np.random.randint(400)
            ind2 = np.random.randint(400)
            if imagePaths[ind1].split(os.path.sep)[-2] != \
                    imagePaths[ind2].split(
                            os.path.sep)[-2]:
                count += 1
                images = []
                image = cv2.imread(imagePaths[ind1])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to
                    # the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                images.append(image)

                image = cv2.imread(imagePaths[ind2])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # check to see if our preprocessors are not None
                if self.preprocessors is not None:
                    # loop over the preprocessors and apply each to
                    # the image
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                images.append(image)

                # treat our processed image as a "feature vector"
                # by updating the data list followed by the labels
                data.append(images)
                labels.append([0])

        # return a tuple of the data and labels
        return (np.array(data), np.array(labels))
