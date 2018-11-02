import numpy as np
import pickle, timeit, random
import os.path
from model import create_model
from logger import logger

log = logger('Feature Extractor')

def embed_image(img):
    try:
        img = (img / 255.).astype(np.float32)
    except:
        return None
    return nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]


start = timeit.default_timer()

nn4_small2 = create_model()

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
#nn4_small2_pretrained.load_weights('weights/nn4.v2.t7')

log.info("Embedding model loaded.")
end = timeit.default_timer()
log.info("Loading CNN embeding model took: %s" % (end-start))