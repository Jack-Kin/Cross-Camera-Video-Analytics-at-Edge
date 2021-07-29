# import numpy as np
# import os
# import urllib.request
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
# from PIL import Image
# from tensorflow.python.platform import gfile
from rknn.api import RKNN


if __name__ == '__main__':

    MODEL_PATH = './yolov3.cfg'
    WEIGHT_PATH = './yolov3.weights'
    RKNN_MODEL_PATH = './yolov3_precompile.rknn'
    DATASET = './dataset.txt'

    # Create RKNN object
    rknn = RKNN()

    NEED_BUILD_MODEL = True

    if NEED_BUILD_MODEL:
        # Load darknet model
        print('--> Loading model')
        ret = rknn.load_darknet(model=MODEL_PATH, weight=WEIGHT_PATH)
        if ret != 0:
            print('Load darknet model failed!')
            exit(ret)
        print('done')

        rknn.config(reorder_channel='2 1 0', channel_mean_value='0 0 0 255')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset='./dataset.txt', pre_compile=True)
        if ret != 0:
            print('Build model failed.')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export RKNN model failed.')
            exit(ret)
        print('done')
    else:
        # Direct load rknn model
        print('Loading RKNN model')
        ret = rknn.load_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Load RKNN model failed.')
            exit(ret)
        print('done')
