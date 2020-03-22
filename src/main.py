import numpy as np
from constants import TREE_PATH, INDEX_DICT_PATH
from model_fn import load_model_handler, predict_image
from search import search
from utility_fn import (
  read_database,
  display_image,
  preprocess_image
)


if __name__ == "__main__":
  loaded_model = load_model_handler('encoder.h5')
  tree, index_dict = read_database(TREE_PATH, INDEX_DICT_PATH)

  test_image_path = list(index_dict.values())[np.random.randint(0,30000)]
  title = test_image_path.split('_')[1].split('.')[0]
  display_image(test_image_path, title)

  test_image = preprocess_image(test_image_path)
  test_latent_features = predict_image(loaded_model, test_image)
  test_latent_features = np.reshape(test_latent_features, (256,))

  nearest_images = search(test_latent_features, tree, index_dict, 5)
  [display_image(item["path"], item["title"]) for item in nearest_images]