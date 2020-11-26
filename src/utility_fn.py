from imutils import paths
from tensorflow.keras.preprocessing.image import (
  img_to_array,
  load_img,
  array_to_img,
)
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import PIL
import random
import vptree as vp

def euclidean(x, y):
  euclidean_distance = np.sqrt(np.sum(np.power(x - y, 2)))
  return euclidean_distance


def preprocess_image(image_path):
  image = load_img(image_path, color_mode='rgb', target_size=(32,32))
  image = img_to_array(image)
  image = image.astype('float32') / 255.0
  image = np.expand_dims(image, axis=0)
  return image


def display_image(image_path, title):
  image = mpimg.imread(image_path)
  plt.figure()
  plt.title(title)
  plt.imshow(image)
  plt.show()


def normalize_latent_features(x):
  return x / np.linalg.norm(x, axis=1, ord=1, keepdims=True)


def construct_tree(points, dist_fn, tree_path):
  print("[INFO] constructing VP Tree")
  tree = vp.VPTree(points, dist_fn)

  print("[INFO] serializing VP Tree")
  with open(tree_path, 'wb') as f:
    f.write(pickle.dumps(tree))


def find_image_paths(dataset_path, length):
  image_paths = list(paths.list_images(dataset_path))
  random.shuffle(image_paths)
  image_paths = image_paths[:length]
  return image_paths


def read_database(tree_path, index_dict_path):
  if tree_path == None:
    index_dict = pickle.loads(open(index_dict_path, 'rb').read())
    return index_dict
  if index_dict_path == None:
    tree = pickle.loads(open(tree_path, 'rb').read())
    return tree
  index_dict = pickle.loads(open(index_dict_path, 'rb').read())
  tree = pickle.loads(open(tree_path, 'rb').read())
  return (tree, index_dict)