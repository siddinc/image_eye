import numpy as np
import pickle
from model_fn import (
  predict_image,
  load_model_handler
)
from constants import (
  BATCH_SIZE,
  INDEX_DICT_PATH,
  TREE_PATH,
)
from utility_fn import (
  preprocess_image,
  euclidean,
  construct_tree,
  read_database,
)


def build_index(model, image_paths):
  count = 0

  for i in range(0, len(image_paths), BATCH_SIZE):
    print("[INFO] processing batch: {}".format(count + 1))
    batch_paths = image_paths[i : i+BATCH_SIZE]
    batch_images = []
    index_list = []

    for j, image_path in enumerate(batch_paths):
      image = preprocess_image(image_path)
      batch_images.append(image)

    batch_images = np.vstack(batch_images)
    latent_features = predict_image(model, batch_images)
    index_list.extend([(batch_path, latent_feature) for batch_path, latent_feature in zip(batch_paths, latent_features)])
    index_dict = {tuple(item[1]): item[0] for item in index_list}

    print("[INFO] serializing Index Dictionary")

    if i == 0:
      pickle.dump(index_dict, open(INDEX_DICT_PATH, 'wb'))
    else:
      loaded_index_dict = pickle.loads(open(INDEX_DICT_PATH, 'rb').read())
      loaded_index_dict.update(index_dict)
      pickle.dump(loaded_index_dict, open(INDEX_DICT_PATH, 'wb'))
    count += 1

  points = list(read_database(None, INDEX_DICT_PATH).keys())
  construct_tree(points, euclidean, TREE_PATH)