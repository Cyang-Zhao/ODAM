import os
import json
import numpy as np
import cv2

color = {'green':(0,255,0),
        'blue':(255,165,0),
        'dark red':(0,0,139),
        'red':(0, 0, 255),
        'dark slate blue':(139,61,72),
        'aqua':(255,255,0),
        'brown':(42,42,165),
        'deep pink':(147,20,255),
        'fuchisia':(255,0,255),
        'yello':(0,238,238),
        'orange':(0,165,255),
        'saddle brown':(19,69,139),
        'black':(0,0,0),
        'white':(255,255,255)}

colors = ['green', 'blue', 'dark red', 'red', 'dark slate blue', 'aqua', 'brown',\
        'deep pink', 'fuchisia', 'yello', 'orange', 'saddle brown','white']
def draw_boxes(img, boxes, scores=None, tags=None, line_thick=1, line_color='white'):
    width = img.shape[1]
    height = img.shape[0]
    for i in range(len(boxes)):
        tmp_color = color[line_color] if line_color is not None else color[colors[i%len(colors)]]
        one_box = boxes[i]
        one_box = np.array([max(one_box[0], 0), max(one_box[1], 0),
                    min(one_box[2], width - 1), min(one_box[3], height - 1)])
        x1,y1,x2,y2 = np.array(one_box[:4]).astype(int)
        cv2.rectangle(img, (x1,y1), (x2,y2), tmp_color, line_thick)
        if scores is not None:
            text = "{} {:.3f}".format(tags[i], scores[i])
            cv2.putText(img, text, (x1, y1 - 7), cv2.FONT_ITALIC, 1.0, tmp_color, line_thick)
    return img

