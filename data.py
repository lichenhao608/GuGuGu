import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(label_file, img_dir):
    '''
    Convert image file into numpy array

    Args:
        label_file: file directory to the csv file containing label info
        img_dir: repository containing images

    Return:
        images: (N, 96, 96) shape image file
        y: (N,) shape label info
    '''

    label = pd.read_csv(label_file)
    y = label['label'].values
    N = len(label)

    images = np.zeros((N, 96, 96))

    for i, name in enumerate(label['id'].values):
        img_file = img_dir + name + '.tif'
        img = plt.imread(img_file)
        # convert image to grayscale
        images[i] = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        if i % 10000 == 0:
            print(f'process {i} images')

    return images, y
