import numpy as np
import cv2
import os
# import urllib.request
# from matplotlib import gridspec
# from matplotlib import pyplot as plt
# from PIL import Image
# from tensorflow.python.platform import gfile
from rknn.api import RKNN
from timeit import default_timer as timer



GRID0 = 13
GRID1 = 26
LISTSIZE = 85
SPAN = 3
NUM_CLS = 80
# MAX_BOXES = 500
OBJ_THRESH = 0.5
NMS_THRESH = 0.6
# OBJ_THRESH = 0.2
# NMS_THRESH = 0.2


CLASSES = ("person", "bicycle", "car","motorbike ","aeroplane ","bus ","train","truck ","boat","traffic light",
           "fire hydrant","stop sign ","parking meter","bench","bird","cat","dog ","horse ","sheep","cow","elephant",
           "bear","zebra ","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife ",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza ","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet ","tvmonitor","laptop	","mouse	","remote ","keyboard ","cell phone","microwave ",
           "oven ","toaster","sink","refrigerator ","book","clock","vase","scissors ","teddy bear ","hair drier", "toothbrush ")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)

    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov3_post_process(input_data):
    # yolov3
    masks = [[3, 4, 5], [0, 1, 2]]
    anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        # x *= image.shape[1]
        # y *= image.shape[0]
        # w *= image.shape[1]
        # h *= image.shape[0]
        # top = max(0, np.floor(x + 0.5).astype(int))
        # left = max(0, np.floor(y + 0.5).astype(int))
        # right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        # bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        # cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        # cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
        #             (top, left - 6),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.6, (0, 0, 255), 2)


def load_model():
    RKNN_MODEL_PATH = './yolov3_precompile.rknn'
    # Create RKNN object
    rknn = RKNN()

    # Direct load rknn model
    print('Loading RKNN model')
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print('load rknn model failed.')
        exit(ret)
    print('done')

    print('--> init runtime')
    ret = rknn.init_runtime()
    # ret = rknn.init_runtime(target='rk1808')
    if ret != 0:
        print('init runtime failed.')
        exit(ret)
    print('done')
    return rknn


if __name__ == '__main__':
    # load and init model
    rknn = load_model()

    accum_time = 0
    curr_fps = 0
    prev_time = timer()
    fps = 'fps: ??'


    try:
        cnt = 1
        while(cnt < 16):
            im_file = './pic/6-' + str(cnt) +'.jpg'
            print(im_file)
            img = cv2.imread(im_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            testtime = timer()
            outputs = rknn.inference(inputs=[img])
            testtime2 = timer()
            print("rknn use time {}", testtime2-testtime)

            input0_data = outputs[0]
            input1_data = outputs[1]
            input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
            input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)
            input_data = []
            input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
            input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
            testtime=timer()
            boxes, classes, scores = yolov3_post_process(input_data)    
            testtime2=timer()
            print("process use time: {}", testtime2-testtime)

            testtime=timer()
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time += exec_time
            curr_fps += 1
            if accum_time > 1:
                accum_time -= 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0    
            if boxes is not None:
                draw(img, boxes, scores, classes)
            else:
                print("No object detected")
            c = cv2.waitKey(5) & 0xff
            if c == 27:
                cv2.destroyAllWindows()
                print("before rknn release")
                rknn.release()
                print("after rknn release")
                break;
            testtime2=timer()
            print("show image use time: {}", testtime2-testtime)
            cnt+=1

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        rknn.release()


    # im_file = './pic/test_416x234.jpg'
    # img = cv2.imread(im_file)
    # # img = cv2.resize(img, (416,416))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # # inference
    # print('--> inference')
    # outputs = rknn.inference(inputs=[img])
    # print('done')

    # input0_data = outputs[0]
    # input1_data = outputs[1]

    # input0_data = input0_data.reshape(SPAN, LISTSIZE, GRID0, GRID0)
    # input1_data = input1_data.reshape(SPAN, LISTSIZE, GRID1, GRID1)

    # input_data = []
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))

    # boxes, classes, scores = yolov3_post_process(input_data)

    # image = cv2.imread(im_file)
    # if boxes is not None:
    #     draw(image, boxes, scores, classes)

    # # cv2.imshow("results", image)
    # # cv2.waitKeyEx(0)

    # rknn.release()
