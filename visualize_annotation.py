import cv2
import glob
import pdb
import numpy as np
from mep import mep

image_list = glob.glob("./data/icdar15/test_images/*")

from tqdm import tqdm


def load_gt(gt_path):
    lines = open(gt_path, encoding='utf-8').readlines()
    bboxes = []
    words = []
    for line in lines:
        ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
        box = [int(ori_box[j]) for j in range(8)]
        word = ori_box[8:]
        word = ','.join(word)
        box = np.array(box, np.int32).reshape(4, 2)
        if word == '###':
            words.append('###')
            bboxes.append(box)
            continue
        area, p0, p3, p2, p1, _, _ = mep(box)

        bbox = np.array([p0, p1, p2, p3])
        distance = 10000000
        index = 0
        for i in range(4):
            d = np.linalg.norm(box[0] - bbox[i])
            if distance > d:
                index = i
                distance = d
        new_box = []
        for i in range(index, index + 4):
            new_box.append(bbox[i % 4])
        new_box = np.array(new_box)
        bboxes.append(np.array(new_box))
        words.append(word)
    return bboxes, words


for i in tqdm(image_list):
    k = i.replace("test_images", "test_gt")
    k = k.replace("img_", "gt_img_")
    k = k.replace("jpg", "txt")

    img = cv2.imread(i)
    bboxes, words = load_gt(k)
    for h, w in zip(bboxes, words):
        if w == '###':
            print(111)
            continue
        h = h.astype(np.int32)
        cv2.polylines(img, [h.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
    cv2.imwrite(i.split("/")[-1], img)



