import os


TREE_PATH = os.path.abspath('../database/tree.cpickle')
INDEX_DICT_PATH = os.path.abspath('../database/index_dict.cpickle')
ENCODER_PATH = os.path.abspath('../models/encoder.h5')
DECODER_PATH = os.path.abspath('../models/decoder.h5')
SAVE_MODEL_PATH = os.path.abspath('../models')
LOAD_MODEL_PATH = os.path.abspath('../models')
AUTOENCODER_PATH = os.path.abspath('../models/autoencoder.h5')
DATASET_PATH = os.path.abspath('../datasets/cifar10')
BATCH_SIZE = 1000
EPOCHS = 20
INIT_LR = 1e-2