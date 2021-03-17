import math
import json
import sys
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
import nrrd
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import pprint
from copy import copy

import torch
#torch.multiprocessing.set_start_method('forkserver')

import pycocotools
import detectron2.structures as structures
import detectron2.data.datasets.coco as coco
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog,\
        build_detection_train_loader,\
        build_detection_test_loader
from detectron2.engine.defaults import DefaultTrainer,\
        default_argument_parser
from detectron2.engine import launch
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.engine import DefaultPredictor
from detectron2.data import Metadata
from detectron2.utils.visualizer import Visualizer

import cv2

from skimage import filters
from skimage.morphology import flood_fill
from random import shuffle

VAL_SCALE_FAC = 0.0

def init_model():
    cfg = get_cfg()
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor = DefaultPredictor(cfg)
    return predictor

def gen_metadata(file_path):
    predictor = init_model()
    im = cv2.imread(file_path)
    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    metadata = Metadata(evaluator_type='coco', image_root='.',
            json_file='',
            name='metadata',
            thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
            thing_dataset_id_to_contiguous_id=
                {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
            )
    output = predictor(im)
    insts = output['instances']
    selector = insts.pred_classes==0
    selector = selector.cumsum(axis=0).cumsum(axis=0) == 1
    results = {}
    for i in range(1, 5):
        temp = insts.pred_classes==i
        selector += temp.cumsum(axis=0).cumsum(axis=0) == 1
    fish = insts[insts.pred_classes==0]
    #print(fish)
    if len(fish):
        results['fish'] = []
        for _ in range(len(fish)):
            results['fish'].append({})
    else:
        fish = None
    results['has_fish'] = bool(fish)
    try:
        ruler = insts[insts.pred_classes==1][0]
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        results['ruler_bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    results['has_ruler'] = bool(ruler)
    try:
        two = insts[insts.pred_classes==3][0]
    except:
        two = None
    try:
        three = insts[insts.pred_classes==4][0]
    except:
        three = None
    if ruler and two and three:
        scale = calc_scale(two, three)
        results['scale'] = scale
    else:
        scale = None
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    vis = visualizer.draw_instance_predictions(insts[selector].to('cpu'))
    os.makedirs('images', exist_ok=True)
    file_name = file_path.split('/')[-1]
    print(file_name)
    cv2.imwrite(f'images/gen_mask_prediction_{file_name}.png',
            vis.get_image()[:, :, ::-1])
    if fish:
        try:
            eyes = insts[insts.pred_classes==2]
        except:
            eyes = None
        for i in range(len(fish)):
            curr_fish = fish[i]
            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                        range(len(eyes))]
                # TODO: Add pred score as a secondary key in event there are
                #       more than one 1.0 overlap eyes
                max_ind = max(range(len(eye_ols)), key=eye_ols.__getitem__)
                eye = eyes[max_ind]
            else:
                eye = None
            results['fish'][i]['has_eye'] = bool(eye)
            results['fish_count'] = len(insts[(insts.pred_classes==0).
                logical_and(insts.scores > 0.3)])

            #try:
            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().
                      numpy().astype('float64')[0]]
            im_crop = im_gray[bbox[1]:bbox[3],bbox[0]:bbox[2]]
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask = gen_mask(bbox, file_path, file_name, im_gray, val,
                    detectron_mask, index=i)
            #except:
                #return {file_name: {'errored': True}}
            if not np.count_nonzero(mask):
                print('Mask failed: {file_name}')
                results['errored'] = True
            else:
                #print(mask)
                im_crop = im_gray[bbox[1]:bbox[3],bbox[0]:bbox[2]].reshape(-1)
                mask_crop = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]].reshape(-1)
                #print(list(zip(list(im_crop),list(mask_crop))))
                #print(np.count_nonzero(mask_crop))
                fground = im_crop[np.where(mask_crop)]
                bground = im_crop[np.where(np.logical_not(mask_crop))]
                #print(im_crop.shape)
                #print(fground.shape)
                #print(bground.shape)
                results['fish'][i]['foreground'] = {}
                results['fish'][i]['foreground']['mean'] = np.mean(fground)
                results['fish'][i]['foreground']['std'] = np.std(fground)
                results['fish'][i]['background'] = {}
                results['fish'][i]['background']['mean'] = np.mean(bground)
                results['fish'][i]['background']['std'] = np.std(bground)
                results['fish'][i]['bbox'] = list(bbox)
                results['fish'][i]['mask'] = mask.astype('uint8').tolist()
                results['fish'][i]['mask'] = '[...]'

                centroid, evec = pca(mask)
                if scale:
                    results['fish'][i]['length'] = fish_length(mask, centroid,
                            evec, scale)
                results['fish'][i]['centroid'] = centroid.tolist()
                if eye:
                    #print(eye.pred_boxes.get_centers())
                    eye_center = [round(x) for x in
                            eye.pred_boxes.get_centers()[0].cpu().numpy()]
                    results['fish'][i]['eye_center'] = list(eye_center)
                    dist1 = distance(centroid, eye_center + evec)
                    dist2 = distance(centroid, eye_center - evec)
                    if dist2 > dist1:
                        #print("HERE")
                        #print(evec)
                        evec *= -1
                        #print(evec)
                    if evec[0] <= 0.0:
                        results['fish'][i]['side'] = 'left'
                    else:
                        results['fish'][i]['side'] = 'right'
                    x_mid = int(bbox[0] + (bbox[2] - bbox[0]) / 2)
                    y_mid = int(bbox[1] + (bbox[3] - bbox[1]) / 2)
                    snout_vec = find_snout_vec(np.array([x_mid, y_mid]), eye_center, mask)
                    if snout_vec is None:
                        results['fish'][i]['clock_value'] =\
                                clock_value(evec,file_name)
                    else:
                        results['fish'][i]['clock_value'] =\
                                clock_value(snout_vec,file_name)
                results['fish'][i]['primary_axis'] = list(evec)
                #print(curr_fish)
                results['fish'][i]['score'] = float(curr_fish.scores[0].cpu())
                #print(results['fish'][i]['score'])
    #pprint.pprint(results)
    return {file_name: results}

def adaptive_threshold(bbox, im_gray):
    #bbox_d = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().
            #numpy().astype('float64')[0]]
    im_crop = im_gray[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    val = filters.threshold_otsu(im_crop) * 1.13
    mask = np.where(im_crop > val, 1, 0).astype(np.uint8)
    #f_bbox_crop = curr_fish.pred_masks[0].cpu().numpy()\
            #[bbox_d[1]:bbox_d[3],bbox_d[0]:bbox_d[2]]
    flat_mask = mask.reshape(-1)
    fground = im_crop.reshape(-1)[np.where(flat_mask)]
    bground = im_crop.reshape(-1)[np.where(np.logical_not(flat_mask))]
    mean_b = np.mean(bground)
    mean_f = np.mean(fground)
    #print(f'b: {mean_b} | f: {mean_f}')
    #flipped = mean_b < mean_f
    flipped = False
    diff = abs(mean_b - mean_f)
    #print(diff)
    #val = (mean_b + mean_f) / 2
    if flipped:
        val -= diff * VAL_SCALE_FAC
    else:
        val += diff * VAL_SCALE_FAC
    val = min(max(1, val), 254)
    return val

def find_snout_vec(centroid, eye_center, mask):
    eye_dir = eye_center - centroid
    x1 = centroid[0]
    y1 = centroid[1]
    #print(centroid)
    #print(eye_center)
    #print(eye_dir)
    max_len = 0
    #fallback = np.array([-1,0])
    max_vec = None
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            #print((x, y))
            if mask[y,x]:
                x2 = x
                y2 = y
                curr_dir = np.array([x2 - x1, y2 - y1])
                curr_eye_dir = np.array([x2 - eye_center[0],
                    y2 - eye_center[1]])
                curr_len = np.linalg.norm(curr_dir)
                if curr_len > max_len:
                    fallback = curr_dir
                    max_len = curr_len
                    if curr_len > np.linalg.norm(curr_eye_dir):
                        max_vec = curr_dir
    #print(max_vec)
    if max_len == 0:
        #return np.array([-1,0])
        return None
    if max_vec is None:
        print(f'Failed snout')
        #max_vec = fallback
        return None
    return max_vec / max_len

def angle(vec1, vec2):
    #print(f'angle: {vec1}, {vec2}')
    return math.acos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def clock_value(evec, file_name):
    if evec[0] < 0:
        if evec[1] < 0:
            #print(file_name)
            comp = np.array([-1,0])
            start = 9
        else:
            comp = np.array([0,-1])
            start = 6
    else:
        if evec[1] < 0:
            comp = np.array([1,0])
            start = 3
        else:
            comp = np.array([0,1])
            start = 0
    ang = angle(comp, evec)
    #print(ang)
    clock = start + (ang / (2 * math.pi) * 12)
    #print(clock)
    if clock > 11.5:
        clock = 12
    elif clock < 0.5:
        clock = 12
    #print(evec)
    return round(clock)

def fish_length(mask, centroid, evec, scale):
    m1 = evec[1] / evec[0]
    m2 = evec[0] / evec[1]
    x1 = centroid[0]
    y1 = centroid[1]
    x_min = centroid[0]
    x_max = centroid[0]
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y,x]:
                x2 = x
                y2 = y
                x_calc = (-y1+y2+m1*x1-m2*x2)/(m1-m2)
                y_calc = m1*(x-x1)+y1
                if x_calc > x_max:
                    x_max = x_calc
                    y_max = y_calc
                elif x_calc < x_min:
                    x_min = x_calc
                    y_min = y_calc
    return distance((x_max, y_max), (x_min, y_min)) / scale

def overlap(fish, eye):
    fish = list(fish.pred_boxes.tensor.cpu().numpy()[0])
    eye = list(eye.pred_boxes.tensor.cpu().numpy()[0])
    if not (fish[0] < eye[2] and eye[0] < fish[2] and fish[1] < eye[3]
            and eye[1] < eye[3]):
        return 0
    pairs = list(zip(fish, eye))
    ol_area = (max(pairs[0]) - min(pairs[2])) * (max(pairs[1]) - min(pairs[3]))
    ol_pct = ol_area / ((eye[0] - eye[2]) * (eye[1] - eye[3]))
    return ol_pct

def pca(img):
    #print(np.count_nonzero(img))
    moments = cv2.moments(img)
    centroid = (int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]))
    #print(centroid)
    y, x = np.nonzero(img)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    if evals[0] > evals[1]:
        evec = evecs[0]
    else:
        evec = evecs[1]

    #sort_indices = np.argsort(evals)[::-1]
    #return (np.array(centroid), evecs[:, sort_indices[0]])
    return (np.array(centroid), evec)

def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def calc_scale(two, three):
    pt1 = two.pred_boxes.get_centers()[0]
    pt2 = three.pred_boxes.get_centers()[0]
    scale = distance([float(pt1[0]), float(pt1[1])],
            [float(pt2[0]), float(pt2[1])])
    scale /= 2.54
    #print(f'Pixels/cm: {scale}')
    return scale

def check(arr, val, flipped):
    if flipped:
        return arr > val
    return arr < val

def gen_mask(bbox, file_path, file_name, im_gray, val, detectron_mask,
        index=0, flipped=False):
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])
    bbox_orig = bbox
    bbox = (l,t,r,b)

    im = Image.open(file_path).convert('L')
    arr2 = np.array(im)
    shape = arr2.shape
    done = False
    im_crop = im_gray[t:b,l:r]
    while not done:
        #print(f'val: {val}')
        done = True
        arr0 = np.array(im.crop(bbox))
        bb_size = arr0.size


        #if val is None:
        #print(val)
        val = filters.threshold_otsu(arr0)
        #print(val)
        #arr1 = np.where(check(arr0, val, flipped), 1, 0).astype(np.uint8)
        arr1 = np.where(arr0 < val, 1, 0).astype(np.uint8)
        indicies = list(zip(*np.where(arr1 == 1)))
        shuffle(indicies)
        count = 0
        for ind in indicies:
            count += 1
            if count > 10000:
                print(f'ERROR on flood fill: {name}')
                return None
            temp = flood_fill(arr1, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / bb_size
            if percent > 0.1:
                for i in (0,temp.shape[0]-1):
                    for j in (0,temp.shape[1]-1):
                        temp = flood_fill(temp, (i, j), 2)
                arr1 = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        #print(np.count_nonzero(arr1))
        arr3 = np.full(shape, 0).astype(np.uint8)
        #print(arr1.shape)
        #print(shape)
        #print(f'{t}:{b},{l}:{r}')
        #print(np.count_nonzero(arr1))
        #print('=====')
        arr3[t:b,l:r] = arr1
        #im_crop = im_gray[t:b,l:r]
        #fground = im_crop.reshape(-1)[arr1.reshape(-1)]
        #bground = im_crop.reshape(-1)[np.invert(arr1.reshape(-1))]
        #mean_b = np.mean(bground)
        #mean_f = np.mean(fground)
        #flipped = mean_b < mean_f
        #print(val)
        #val = (mean_b + mean_f) / 2
        #print(val)
        #if flipped:
        #    val -= val * VAL_SCALE_FAC
        #else:
        #    val += val * VAL_SCALE_FAC
        #val = min(max(1, val), 254)
        try:
            if np.any(arr3[t:b,l] != 0) and l > 0:
                l -= 1
                l = max(0, l)
                done = False
            if np.any(arr3[t:b,r] != 0) and r < shape[1] - 1:
                r += 1
                r = min(shape[1] - 1, r)
                done = False
            if np.any(arr3[t,l:r] != 0) and t > 0:
                t -= 1
                t = max(0, t)
                done = False
            if np.any(arr3[b,l:r] != 0) and b < shape[0] - 1:
                b += 1
                b = min(shape[0] - 1, b)
                done = False
        except:
            print(f'{file_name}: Error expanding bounding box')
            done = True
        bbox = (l,t,r,b)
        #val = adaptive_threshold(bbox, im_gray)
    #print(list(arr1.reshape(-1)))
    if np.count_nonzero(arr1) / bb_size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        arr3 = detectron_mask.astype('uint8')
        bbox = bbox_orig
    arr4 = np.where(arr3 == 1, 255, 0).astype(np.uint8)
    (l,t,r,b) = shrink_bbox(arr3)
    arr4[t:b,l] = 175
    arr4[t:b,r] = 175
    arr4[t,l:r] = 175
    arr4[b,l:r] = 175
    im2 = Image.fromarray(arr4, 'L')
    im2.save(f'images/gen_mask_mask_{file_name}_{index}.png')
    return (bbox, arr3)

#https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
def shrink_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    #print(mask)
    #print(rows)
    #print(cols)
    #exit(0)
    #try:
    #print(np.where(cols))
    #print()
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    #except:
        #return None

    return (cmin, rmin, cmax, rmax)

def main():
    direct = sys.argv[1]
    if os.path.isdir(direct):
        files = [entry.path for entry in os.scandir(direct)][:1000]
    else:
        files = [direct]
    #print(files)
    #predictor = init_model()
    #f = partial(gen_metadata, predictor)
    with Pool(4) as p:
        #results = map(gen_metadata, files)
        results = p.map(gen_metadata, files)
    #results = map(gen_metadata, files)
    output = {}
    for i in results:
        output[list(i.keys())[0]] = list(i.values())[0]
    #print(output)
    if len(output) > 1:
        with open('metadata.json', 'w') as f:
            json.dump(output, f)
    else:
        pprint.pprint(output)

if __name__ == '__main__':
    #gen_metadata(sys.argv[1])
    main()
