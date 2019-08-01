# coding: utf-8
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import cv2
import sys
import time
from utils.class_labels import *
from utils.data_process import letterbox_resize, parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from darknet import Darknet
from utils import detection_boxes_tensorflow as vis
import PySimpleGUI as sg


def arg_parse():
    i_vid = r'assets/cars.mp4'
    y_path = r'config/coco.names'
    anchor_path = r'config/yolo_anchors.txt'
    ckpt_path = r'weight/yolov3.ckpt'
    layout = [
        [sg.Text('Tensorflow-YOLO Video Player', size=(18, 1), font=('Any', 18), text_color='#1c86ee', justification='left')],
        [sg.Text('Path to input video'), sg.In(i_vid, size=(40, 1), key='video'), sg.FileBrowse()],
        [sg.Text('Path to ckpt File'), sg.In(ckpt_path, size=(40, 1), key='ckpt'), sg.FileBrowse()],
        [sg.Text('Path to anchor File'), sg.In(anchor_path, size=(40, 1), key='anchor_input'), sg.FileBrowse()],
        [sg.Text('Path to label'), sg.In(y_path, size=(40, 1), key='label'), sg.FolderBrowse()],
        [sg.Text('Confidence'),
         sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.5, size=(15, 15), key='confidence')],
        [sg.Text('NMSThreshold'),
         sg.Slider(range=(0, 1), orientation='h', resolution=.1, default_value=.4, size=(15, 15), key='nms_threshold')],
        [sg.Text('Resolution'), sg.Radio('320', "resolution", key="small_resol"),
         sg.Radio('416', "resolution", default=True, key="best_resol"),
         sg.Radio('512', "resolution", key="large_resol")],
        [sg.Text("Classes not to detect"), sg.Listbox(values=class_names, default_values=class_names,
                                                      select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, size=(30, 10),
                                                      key='class_list')],
        [sg.Text(' ' * 8), sg.Checkbox('Use webcam', key='webcam')],
        [sg.OK(), sg.Cancel()]
    ]
    win = sg.Window('YOLO Video',
                    default_element_size=(14, 1),
                    text_justification='right',
                    auto_size_text=False).Layout(layout)
    event, values = win.Read()

    if event is None or event == 'Cancel':
        exit()
    args = values

    win.Close()

    return args


def run_inference_for_single_image(frame, lbox_resize, sess, input_data, inp_dim, boxes, scores, labels):
    if lbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(frame, inp_dim, inp_dim)
    else:
        height_ori, width_ori = frame.shape[:2]
        img = cv2.resize(frame, tuple([inp_dim, inp_dim]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

    # rescale the coordinates to the original image
    if lbox_resize:
        boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
        boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
    else:
        boxes_[:, [0, 2]] *= (width_ori / float(inp_dim))
        boxes_[:, [1, 3]] *= (height_ori / float(inp_dim))

    return boxes_, scores_, labels_


def main():
    args = arg_parse()

    # anchors and class labels
    anchors = parse_anchors(args['anchor_input'])
    classes = read_class_names(args['label'])
    num_classes = len(classes)

    VIDEO_PATH = args['video'] if not args['webcam'] else 0

    resolution_list = [320, 416, 512]
    index = [args['small_resol'], args['best_resol'], args['large_resol']].index(True)
    inp_dim = resolution_list[index]

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, inp_dim, inp_dim, 3], name='input_data')
        model = Darknet(num_classes, anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_classes,
                                        max_boxes=200, score_thresh=args['confidence'],
                                        nms_thresh=args['nms_threshold'])

        saver = tf.train.Saver()
        saver.restore(sess, args['ckpt'])

        # Set window
        winName = 'YOLO-Tensorflow'

        try:
            # Read Video file
            cap = cv2.VideoCapture(VIDEO_PATH)
        except IOError:
            print("Input video file", VIDEO_PATH, "doesn't exist")
            sys.exit(1)

        while cap.isOpened():
            hasFrame, frame = cap.read()
            if not hasFrame:
                break
            # Actual Detection
            start = time.time()

            boxes_, scores_, labels_ = run_inference_for_single_image(frame,
                                                                      False,
                                                                      # letterbox_resize,
                                                                      sess,
                                                                      input_data,
                                                                      inp_dim,
                                                                      boxes,
                                                                      scores,
                                                                      labels)
            # Visualization of the results of a detection
            vis.visualize_boxes_and_labels_yolo(frame,
                                                boxes_,
                                                classes,
                                                labels_,
                                                scores_,
                                                args['class_list'],
                                                use_normalized_coordinates=False)

            end = time.time()

            cv2.putText(frame, '{:.2f}ms'.format((end - start) * 1000), (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
            cv2.imshow(winName, frame)
            print("FPS {:5.2f}".format(1 / (end - start)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        print("Video ended")

        # releases video and removes all windows generated by the program
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
