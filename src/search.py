import numpy as np
from utility_fn import (
  euclidean,
  display_image
)


def search(query_image_latent_features, tree, index_dict, limit):
  nearest_images = []
  result = tree.get_n_nearest_neighbors(query_image_latent_features, limit)
  result.sort(key=lambda e: e[0])

  for i,j in result:
    path = index_dict[tuple(j)]
    title = path.split('_')[1].split('.')[0]
    eu_dist = i
    nearest_images.append({
      "path": path,
      "title": title,
      "eu_dist": eu_dist
    })
  return nearest_images