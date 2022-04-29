#!/usr/local/bin/python
'''
This is Kevin Karnani code, modified by thibault tabarin for containerization

'''
import json
import math
import os
import pprint
import sys
import yaml
from random import shuffle

import gc
import torch
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2.data import Metadata
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, pairwise_iou, pairwise_ioa
from matplotlib import pyplot as plt
from scipy import stats
from skimage import filters, measure
from skimage.morphology import flood_fill
from torch.multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")
# torch.multiprocessing.set_start_method('forkserver')

# ensure the look at the right place for the configuration file
root_file_path = os.path.dirname(__file__)

VAL_SCALE_FAC = 0.5
conf = json.load(open(os.path.join(root_file_path,'config/config.json'), 'r'))
ENHANCE = bool(conf['ENHANCE'])
JOEL = bool(conf['JOEL'])
PROCESSOR = conf['PROCESSOR']
IOU_PCT = .02

with open(os.path.join(root_file_path,'config/mask_rcnn_R_50_FPN_3x.yaml'), 'r') as f:
    iters = yaml.load(f, Loader=yaml.FullLoader)["SOLVER"]["MAX_ITER"]


def init_model(enhance_contrast=ENHANCE, joel=JOEL, processor=PROCESSOR):
    """
    Initialize model using config files for RCNN, the trained weights, and other parameters.

    Returns:
        predictor -- DefaultPredictor(**configs).
    """
    root_file_path = os.path.dirname(__file__)
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join(root_file_path,'config/mask_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    OUTPUT_DIR = os.path.join(root_file_path, 'output')
    if not joel:
        OUTPUT_DIR += f"/non_enhanced" if not enhance_contrast else f"/enhanced"
        # cfg.OUTPUT_DIR += f"/non_enhanced_{iters}" if not enhance_contrast else f"/enhanced_{iters}"
    cfg.MODEL.WEIGHTS = os.path.join(OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = processor
    predictor = DefaultPredictor(cfg)

    return predictor


def gen_metadata(file_path, enhance_contrast=ENHANCE, visualize=False, multiple_fish=False):
    """
    Generates metadata of an image and stores attributes into a Dictionary.

    Parameters:
        file_path -- string of path to image file.
    Returns:
        {file_name: results} -- dictionary of file and associated results.
    """
    predictor = init_model()
    im = cv2.imread(file_path)
    im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if enhance_contrast:
        lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

        # -----Splitting the LAB image to different channels-------------------------
        l, a, b = cv2.split(lab)

        # -----Applying CLAHE to L-channel-------------------------------------------
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
        limg = cv2.merge((cl, a, b))

        # -----Converting image from LAB Color model to RGB model--------------------
        im = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        im_gray = clahe.apply(im_gray)
    metadata = Metadata(evaluator_type='coco', image_root='.',
                        json_file='',
                        name='metadata',
                        thing_classes=['fish', 'ruler', 'eye', 'two', 'three'],
                        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                        )
    output = predictor(im)
    insts = output['instances']
    selector = insts.pred_classes == 0
    selector = selector.cumsum(axis=0).cumsum(axis=0) == 1
    results = {}
    file_name = file_path.split('/')[-1]
    for i in range(1, 5):
        temp = insts.pred_classes == i
        selector += temp.cumsum(axis=0).cumsum(axis=0) == 1
    fish = insts[insts.pred_classes == 0]
    if len(fish):
        results['fish'] = []
        if not multiple_fish:
            results['fish'].append({})
        else:
            for _ in range(len(fish)):
                results['fish'].append({})
    else:
        fish = None
    results['has_fish'] = bool(fish)
    try:
        ruler = insts[insts.pred_classes == 1][0]
        ruler_bbox = list(ruler.pred_boxes.tensor.cpu().numpy()[0])
        results['ruler_bbox'] = [round(x) for x in ruler_bbox]
    except:
        ruler = None
    results['has_ruler'] = bool(ruler)
    try:
        two = insts[insts.pred_classes == 3][0]
    except:
        two = None
    try:
        three = insts[insts.pred_classes == 4][0]
    except:
        three = None
    if ruler and two and three:
        scale = calc_scale(two, three, file_name)
        results['scale'] = scale
        results['unit'] = 'cm'
    else:
        scale = None
    visualizer = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
    vis = visualizer.draw_instance_predictions(insts.to('cpu'))
    f_name = file_name.split('.')[0]
    if visualize:
        cv2.imshow('prediction', np.array(vis.get_image()[:, :, ::-1], dtype=np.uint8))
        cv2.waitKey(0)

    if False:    # to save visualization of the prediction
        os.makedirs('images', exist_ok=True)
        os.makedirs('images/enhanced', exist_ok=True)
        os.makedirs('images/non_enhanced', exist_ok=True)
        dirname = 'images/'
        dirname += 'enhanced/' if enhance_contrast else 'non_enhanced/'
        print(file_name)
        cv2.imwrite(f'{dirname}/gen_prediction_{f_name}.png',
                    vis.get_image()[:, :, ::-1])

    skippable_fish = []
    fish_length = 0
    if fish:
        try:
            eyes = insts[insts.pred_classes == 2]
        except:
            eyes = None

        fish = fish[fish.scores > .3]
        fish_length = len(fish)
        if not multiple_fish:
            fish = fish[fish.scores.argmax().item()]
        for i in range(len(fish)):
            curr_fish = fish[i]
            if multiple_fish:
                if i in skippable_fish:
                    continue
                fish_ols = [overlap_fish(curr_fish, fish[j]) for j in range(i + 1, len(fish))]
                for j in range(len(fish_ols)):
                    if i + j + 1 not in skippable_fish and fish_ols[j] > IOU_PCT:
                        results['fish'].pop(i + j + 1 - len(skippable_fish))
                        skippable_fish.append(i + j + 1)
                    else:
                        print(f"Fish {i} and Fish {i + j + 1} do not overlap!")
            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                           range(len(eyes))]
                eye = None
                if not all(ol == 0 for ol in eye_ols):
                    full = [i for i in range(
                        len(eye_ols)) if eye_ols[i] >= .95]

                    # if multiple eyes with 95% or greater overlap, pick highest confidence
                    if len(full) > 1:
                        eye = eyes[full]
                        eye = eye[eye.scores.argmax().item()]
                    else:
                        max_ind = max(range(len(eye_ols)),
                                      key=eye_ols.__getitem__)
                        eye = eyes[max_ind]
            else:
                eye = None
            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0]]
            need_scaling = False
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask, pixel_anal_failed = gen_mask(bbox, file_path,
                                                     file_name, im_gray, val, detectron_mask)

            mask_uint8 = np.where(mask == 1, 255, 0).astype(np.uint8)

            if False: # Activate with True to save the upgraded mask generate by gen_mask()
                os.makedirs('mask', exist_ok=True)
                dirname = 'mask/'
                print("save mask for",file_name)
                cv2.imwrite(f'{dirname}/mask_{f_name}.png',mask_uint8)

            centroid, evecs, cont_length, cont_width, length, width, area = pca(mask, scale)
            major, minor = evecs[0], evecs[1]

            if not np.count_nonzero(mask):
                print('Mask failed: {file_name}')
                results['errored'] = True
            else:
                im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                mask_crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].reshape(-1)
                mask_coords = np.argwhere(mask != 0)[:, [1, 0]]
                fground = im_crop[np.where(mask_crop)]
                bground = im_crop[np.where(np.logical_not(mask_crop))]
                results['fish'][i]['foreground'] = {}
                results['fish'][i]['foreground']['mean'] = np.mean(fground)
                results['fish'][i]['foreground']['std'] = np.std(fground)
                results['fish'][i]['background'] = {}
                results['fish'][i]['background']['mean'] = np.mean(bground)
                results['fish'][i]['background']['std'] = np.std(bground)
                results['fish'][i]['bbox'] = list(bbox)
                results['fish'][i]['pixel_analysis_failed'] = pixel_anal_failed
                start, code = encoded_mask(mask)
                region = measure.regionprops(mask)[0]
                if visualize:
                    fig, ax = plt.subplots()
                    ax.imshow(mask, cmap=plt.cm.gray)
                    y0, x0 = region.centroid
                    orientation = region.orientation
                    x1 = x0 + math.cos(orientation) * 0.5 * \
                         region.axis_minor_length
                    y1 = y0 - math.sin(orientation) * 0.5 * \
                         region.axis_minor_length
                    x2 = x0 - math.sin(orientation) * 0.5 * \
                         region.axis_major_length
                    y2 = y0 - math.cos(orientation) * 0.5 * \
                         region.axis_major_length

                    ax.plot((x0, x1), (y0, y1), '-r')
                    ax.plot((x0, x2), (y0, y2), '-b')
                    ax.plot(x0, y0, '.g', markersize=15)

                    minr, minc, maxr, maxc = region.bbox
                    bx = (minc, maxc, maxc, minc, minc)
                    by = (minr, minr, maxr, maxr, minr)
                    ax.plot(bx, by, '-b', linewidth=2.5)
                    plt.show()

                results['fish'][i]['extent'] = region.extent
                results['fish'][i]['eccentricity'] = region.eccentricity
                results['fish'][i]['solidity'] = region.solidity
                results['fish'][i]['skew'] = list(stats.skew(mask_coords))
                results['fish'][i]['kurtosis'] = list(
                    stats.kurtosis(mask_coords))
                results['fish'][i]['std'] = list(np.std(mask_coords, axis=0))
                results['fish'][i]['mask'] = {}
                results['fish'][i]['mask']['start_coord'] = list(start)
                results['fish'][i]['mask']['encoding'] = code

                # upscale fish and then rerun
                if eye is None:
                    need_scaling = True
                    factor = 4
                    eye_center, side, clock_val = upscale(
                        im, bbox, f_name, factor)
                    if eye_center is not None and side is not None:
                        results['fish'][i]['eye_center'] = eye_center
                        results['fish'][i]['side'] = side
                        results['fish'][i]['clock_value'] = clock_val
                        eye = 1  # placeholder, change to something more useful
                if scale:
                    results['fish'][i]['cont_length'] = cont_length
                    results['fish'][i]['cont_width'] = cont_width
                    results['fish'][i]['area'] = area
                    results['fish'][i]['feret_diameter_max'] = region.feret_diameter_max / scale
                    results['fish'][i]['major_axis_length'] = region.major_axis_length / scale
                    results['fish'][i]['minor_axis_length'] = region.minor_axis_length / scale
                    results['fish'][i]['convex_area'] = region.convex_area / \
                                                        (scale ** 2)
                    results['fish'][i]['perimeter'] = measure.perimeter(
                        mask, neighbourhood=8) / scale
                    results['fish'][i]['oriented_length'] = length / scale
                    results['fish'][i]['oriented_width'] = width / scale
                results['fish'][i]['centroid'] = centroid.tolist()
            results['fish'][i]['has_eye'] = bool(eye)
            if eye and not need_scaling:
                eye_center = [round(x) for x in eye.pred_boxes.get_centers()[0].cpu().numpy()]
                results['fish'][i]['eye_center'] = list(eye_center)
                dist1 = distance(centroid, eye_center + major)
                dist2 = distance(centroid, eye_center - major)
                if dist2 > dist1:
                    major *= -1
                if major[0] <= 0.0:
                    results['fish'][i]['side'] = 'left'
                else:
                    results['fish'][i]['side'] = 'right'
                snout_vec = major
                if snout_vec is None:
                    results['fish'][i]['clock_value'] = \
                        clock_value(major, file_name)
                else:
                    results['fish'][i]['clock_value'] = \
                        clock_value(snout_vec, file_name)
                results['fish'][i]['primary_axis'] = list(major)
                results['fish'][i]['score'] = float(curr_fish.scores[0].cpu())
    results['fish_count'] = len(insts[(insts.pred_classes == 0).logical_and(insts.scores > 0.3)]) - \
                            len(skippable_fish) if multiple_fish else int(results['has_fish'])
    results['detected_fish_count'] = fish_length
    return {f_name: results}, mask_uint8


def gen_metadata_upscale(file_path, fish):
    gc.collect()
    torch.cuda.empty_cache()
    predictor = init_model()
    im = fish
    im_gray = cv2.cvtColor(fish, cv2.COLOR_BGR2GRAY)
    output = predictor(im)
    insts = output['instances']
    selector = insts.pred_classes == 0
    selector = selector.cumsum(axis=0).cumsum(axis=0) == 1
    results = {}
    file_name = file_path.split('/')[-1]
    f_name = file_name.split('.')[0]
    for i in range(1, 5):
        temp = insts.pred_classes == i
        selector += temp.cumsum(axis=0).cumsum(axis=0) == 1
    fish = insts[insts.pred_classes == 0]
    if len(fish):
        results['fish'] = []
        results['fish'].append({})
    else:
        fish = None
    results['has_fish'] = bool(fish)
    if fish:
        try:
            eyes = insts[insts.pred_classes == 2]
        except:
            eyes = None

        fish = fish[fish.scores > .3]
        fish = fish[fish.scores.argmax().item()]
        for i in range(len(fish)):
            curr_fish = fish[i]
            if eyes:
                eye_ols = [overlap(curr_fish, eyes[j]) for j in
                           range(len(eyes))]
                eye = None
                if not all(ol == 0 for ol in eye_ols):
                    full = [i for i in range(
                        len(eye_ols)) if eye_ols[i] >= .95]

                    # if multiple eyes with 95% or greater overlap, pick highest confidence
                    if len(full) > 1:
                        eye = eyes[full]
                        eye = eye[eye.scores.argmax().item()]
                    else:
                        max_ind = max(range(len(eye_ols)),
                                      key=eye_ols.__getitem__)
                        eye = eyes[max_ind]
            else:
                eye = None
            bbox = [round(x) for x in curr_fish.pred_boxes.tensor.cpu().numpy().astype('float64')[0]]
            detectron_mask = curr_fish.pred_masks[0].cpu().numpy()
            val = adaptive_threshold(bbox, im_gray)
            bbox, mask, pixel_anal_failed = gen_mask_upscale(bbox, file_path,
                                                             file_name, im_gray, val, detectron_mask)
            centroid, evecs = pca(mask)[:2]
            major, minor = evecs[0], evecs[1]
            results['fish'][i]['has_eye'] = bool(eye)
            if eye:
                eye_center = [round(x) for x in eye.pred_boxes.get_centers()[0].cpu().numpy()]
                results['fish'][i]['eye_center'] = list(eye_center)
                dist1 = distance(centroid, eye_center + major)
                dist2 = distance(centroid, eye_center - major)
                if dist2 > dist1:
                    major *= -1
                results['fish'][i]['side'] = 'left' if major[0] <= 0.0 else 'right'
                snout_vec = major
                results['fish'][i]['clock_value'] = clock_value(major if snout_vec is None else snout_vec, file_name)
    return {f_name: results}


def upscale(im, bbox, f_name, factor):
    h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
    scaled = cv2.resize(im[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy(), (w * factor, h * factor),
                        interpolation=cv2.INTER_CUBIC)
    os.makedirs('images/testing', exist_ok=True)
    cv2.imwrite(f'images/testing/{f_name}.png', scaled)
    eye_center, side, clock_val, scale = None, None, None, None
    new_data = gen_metadata_upscale(f'images/testing/{f_name}.png', scaled)
    if 'fish' in new_data[f'{f_name}'] and new_data[f'{f_name}']['fish'][0]['has_eye']:
        eye_center = new_data[f'{f_name}']['fish'][0]['eye_center']
        eye_x, eye_y = eye_center
        eye_y //= factor
        eye_y += bbox[1]
        eye_x //= factor
        eye_x += bbox[0]
        eye_center = [eye_x, eye_y]
        side = new_data[f'{f_name}']['fish'][0]['side']
        clock_val = new_data[f'{f_name}']['fish'][0]['clock_value']
    if os.path.isfile(f'images/testing/{f_name}.png'):
        os.remove(f'images/testing/{f_name}.png')
    return eye_center, side, clock_val


def adaptive_threshold(bbox, im_gray):
    """
    Determines the best thresholding value.
    Parameters:
        bbox -- bounding box in [top left x, top left y, bottom right x, bottom right y] format.
        im_gray -- grayscale version of original image.
    Returns:
        val -- new threshold.
    """
    im_crop = im_gray[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    val = filters.threshold_otsu(im_crop)
    mask = np.where(im_crop > val, 1, 0).astype(np.uint8)
    flat_mask = mask.reshape(-1)
    bground = im_crop.reshape(-1)[np.where(np.logical_not(flat_mask))]
    mean_b = np.mean(bground)
    flipped = False
    diff = abs(mean_b - val)
    if flipped:
        val -= diff * VAL_SCALE_FAC
    else:
        val += diff * VAL_SCALE_FAC
    val = min(max(1, val), 254)
    return val


def find_snout_vec(centroid, eye_center, mask):
    """
    Determine the direction of the snout.
    Parameters:
        centroid -- center of fish in [x, y] format.
        eye_center -- center of eye in [x, y] format.
        mask -- thresholded image.
    Returns:
        max_vec / max_len -- vector pointing in direction of snout.
    """
    eye_dir = eye_center - centroid
    x1 = centroid[0]
    y1 = centroid[1]
    max_len = 0
    max_vec = None
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            if mask[y, x]:
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
    if max_len == 0:
        return None
    if max_vec is None:
        print(f'Failed snout')
        return None
    return max_vec / max_len


def angle(vec1, vec2):
    """
    Finds angle between two vectors.
    """
    # print(f'angle: {vec1}, {vec2}')
    return math.acos(vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def clock_value(evec, file_name):
    """
    Creates a clock value depending on the major axis provided.
    Parameters:
        evec -- Eigenvector that depicts the major axis.
        file_name -- path to image file.
    Returns:
        round(clock) -- rounded off clock value, ranging from 1-12.
    """
    if evec[0] < 0:
        if evec[1] > 0:
            comp = np.array([-1, 0])
            start = 9
        else:
            comp = np.array([0, -1])
            start = 6
    else:
        if evec[1] < 0:
            comp = np.array([1, 0])
            start = 3
        else:
            comp = np.array([0, 1])
            start = 0
    ang = angle(comp, evec)
    clock = start + (ang / (2 * math.pi) * 12)
    if clock > 11.5:
        clock = 12
    elif clock < 0.5:
        clock = 12
    return round(clock)


def fish_box_length(mask, centroid, evec, scale):
    """
    Check how far fish pixels gets in each direction from the centroid of
    the fish blob then return fish length. This is done by
    intersection the major axis with a line defined by a given fish pixel
    and the minor axis, then finding which two intersection points are
    farthest from the centroid in each direction.
    Parameters:
        mask -- thresholded image.
        centroid -- center of fish in [x, y] format.
        evec -- major axis of fish.
        scale -- pixels per unit.
    Returns:
        distance -- distance from max to min points on major axis.
    """
    m1 = evec[1] / evec[0]
    m2 = evec[0] / evec[1]
    # Set these as the first point for point slope form of a line
    # to be used with m1
    x1 = centroid[0]
    y1 = centroid[1]
    # Initial values for how far from the major axis
    # points project in each direction
    x_min = centroid[0]
    x_max = centroid[0]
    # Loop over every pixel in the bounding box
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            # If it is a fish pixel
            if mask[y, x]:
                # Set this as the second point for point slope form of a line
                # to be sued with m2
                x2 = x
                y2 = y
                # Intersect the major axis with the line formed by x2, y2 and
                # m2. I calculated this using basic algebra given the two
                # line equations.
                x_calc = (-y1 + y2 + m1 * x1 - m2 * x2) / (m1 - m2)
                y_calc = m1 * (x_calc - x1) + y1
                # If this is the new furthest point in one or the other,
                # save it
                if x_calc > x_max:
                    x_max = x_calc
                    y_max = y_calc
                elif x_calc < x_min:
                    x_min = x_calc
                    y_min = y_calc
    # Return the distance between the points we've found scaled into cms
    return distance((x_max, y_max), (x_min, y_min)) / scale


def overlap(fish, eye):
    """
    Checks if the eye is in the fish.
    Parameters:
        fish -- fish coordinates.
        eye -- eye coordinates.
    Returns:
        ol_pct -- percent of eye that is inside the fish.
    """
    fish = list(fish.pred_boxes.tensor.cpu().numpy()[0])
    eye = list(eye.pred_boxes.tensor.cpu().numpy()[0])
    if not (fish[0] < eye[2] and eye[0] < fish[2] and fish[1] < eye[3]
            and eye[1] < eye[3]):
        return 0
    pairs = list(zip(fish, eye))
    ol_area = (max(pairs[0]) - min(pairs[2])) * (max(pairs[1]) - min(pairs[3]))
    ol_pct = ol_area / ((eye[0] - eye[2]) * (eye[1] - eye[3]))
    return ol_pct


def overlap_eye(fish, eye):
    """
    Checks if the fish overlaps with the eye.
    """
    fish = Boxes(fish.pred_boxes.tensor)
    eye = Boxes(eye.pred_boxes.tensor)
    return pairwise_ioa(fish, eye).item()


def overlap_fish(fish1, fish2):
    """
    Checks if the two fish overlap.
    """
    fish1 = Boxes(fish1.pred_boxes.tensor)
    fish2 = Boxes(fish2.pred_boxes.tensor)
    return pairwise_iou(fish1, fish2).item()


# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
def pca(img, glob_scale=None, visualize=False):
    """
    Performs principle component analysis on a grayscale image.
    Parameters:
        img -- grayscale image.
        glob_scale -- pixels per unit.
    Returns:
        np.array(centroid) -- numpy array containing centroid.
        evecs[:, sort_indices[0]] -- major axis, or eigenvector associated with highest eigenvalue.
        length -- length of fish.
        width -- width of fish.
        area -- area of fish.
    """
    moments = cv2.moments(img)
    centroid = (int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]))
    y, x = np.nonzero(img)

    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])

    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    # Eigenvector with largest eigenvalue
    x_v1, y_v1 = evecs[:, sort_indices[0]]
    # negate eigenvector
    if x_v1 < 0:
        x_v1 *= -1
        y_v1 *= -1
    theta = np.arctan2(y_v1, x_v1)
    rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                              [np.sin(theta), np.cos(theta)]])
    transformed_mat = rotation_mat * coords
    x_transformed, y_transformed = transformed_mat.A
    x_round, y_round = x_transformed.round(
        decimals=0), y_transformed.round(decimals=0)
    x_vals, x_counts = np.unique(x_round, return_counts=True)
    y_vals, y_counts = np.unique(y_round, return_counts=True)
    x_calc, y_calc = x_vals[x_counts.argmax()], y_vals[y_counts.argmax()]
    x_indices, y_indices = np.where(
        x_round == x_calc), np.where(y_round == y_calc)
    cont_width = y_round[x_indices].max() - y_round[x_indices].min()
    cont_length = x_round[y_indices].max() - x_round[y_indices].min()
    width = y_vals.max() - y_vals.min()
    length = x_vals.max() - x_vals.min()

    if visualize:
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        scale = 300
        plt.plot([x_v1 * -scale * 2, x_v1 * scale * 2],
                 [y_v1 * -scale * 2, y_v1 * scale * 2], color='red')
        plt.plot([x_v2 * -scale, x_v2 * scale],
                 [y_v2 * -scale, y_v2 * scale], color='blue')
        plt.plot(x, y, 'y.')
        plt.axis('equal')
        plt.gca().invert_yaxis()  # Match the image system with origin at top left
        plt.axhline(y=y_calc)
        plt.axvline(x=x_calc)
        plt.plot(x_transformed, y_transformed, 'g.')
        plt.show()

    area = transformed_mat.shape[1]
    if glob_scale is not None:
        cont_length /= glob_scale
        cont_width /= glob_scale
        length /= glob_scale
        width /= glob_scale
        area /= glob_scale ** 2

    return np.array(centroid), evecs[:, sort_indices], cont_length, cont_width, length, width, area


def find_nearest(array, value):
    """
    Find the nearest element of array to the given value
    """
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def encode_freeman(image_contour):
    """
    Encode the image contour in an 8-direction freeman chain code based on angles
    """
    freeman_code = ""
    freeman_dict = {-90: '0', -45: '1', 0: '2',
                    45: '3', 90: '4', 135: '5', 180: '6', -135: '7'}
    allowed_directions = np.array([0, 45, 90, 135, 180, -45, -90, -135])

    for i in range(len(image_contour) - 1):
        delta_x = image_contour[i + 1][1] - image_contour[i][1]
        delta_y = image_contour[i + 1][0] - image_contour[i][0]
        angle = allowed_directions[np.abs(
            allowed_directions - np.rad2deg(np.arctan2(delta_y, delta_x))).argmin()]
        if not (delta_x == 0 and delta_y == 0):
            freeman_code += freeman_dict[angle]

    return freeman_code


def create_svg(contour, shape):
    with open('image.svg', 'w+') as f:
        f.write(
            f'<svg width="{shape[1]}" height="{shape[0]}" xmlns="http://www.w3.org/2000/svg">')
        f.write('<path d="M')
        for coords in contour:
            x, y = coords
            f.write(f"{int(x)} {int(y)} ")
        f.write('" stroke="red" fill="none"/>')
        f.write('</svg>')


def encoded_mask(mask, visualize=False):
    # Extract the longest contour in the image
    contours = measure.find_contours(mask, 0.9)
    contours_main = np.around(max(contours, key=len), decimals=0)

    if visualize:
        # Display the image and plot the main contour found
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap=plt.cm.gray)
        ax.plot(contours_main[:, 1], contours_main[:, 0])
    # a = encode_freeman(contours_main)
    # b = decode_freeman(contours_main, mask, a)
    # Extract freeman code from contour
    return contours_main[0][::-1], encode_freeman(contours_main)


def decode_freeman(contour, mask, code, visualize=False):
    coords = [list(contour[0][::-1])]
    freeman_dict = {0: [0, -1], 1: [1, -1], 2: [1, 0],
                    3: [1, 1], 4: [0, 1], 5: [-1, 1], 6: [-1, 0], 7: [-1, -1]}
    for letter in code:
        change = freeman_dict[int(letter)]
        current = coords[-1]
        coords.append([current[0] + change[0], current[1] + change[1]])
    # create_svg(coords, mask.shape)
    # np.savetxt('foo.csv', coords, delimiter=",", fmt='%f')
    if visualize:
        cnt = np.array(coords)
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap=plt.cm.gray)
        ax.plot(cnt[:, 0], cnt[:, 1])
        plt.show()
    return coords


def perimeter(code, scale):
    even_numbers = ''.join(filter(lambda x: int(x) % 2 == 0, list(code)))
    odd_numbers = ''.join(filter(lambda x: int(x) % 2 == 1, list(code)))
    return (len(even_numbers) + np.sqrt(2) * len(odd_numbers)) / scale


def distance(pt1, pt2):
    """
    Returns the 2-D Euclidean Distance between 2 points.
    """
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def calc_scale(two, three, file_name):
    """
    Calculates the pixels per unit.
    Parameters:
        two -- the "two" from the ruler in the image.
        three -- the "three" from the ruler in the image.
        file_name -- name of Image in file path.
    Returns:
        scale -- pixels between the centers of the "two" and "three".
    """
    cm_list = ['uwzm']
    in_list = ['inhs']
    file_name = file_name.lower()
    pt1 = two.pred_boxes.get_centers()[0]
    pt2 = three.pred_boxes.get_centers()[0]
    scale = distance([float(pt1[0]), float(pt1[1])],
                     [float(pt2[0]), float(pt2[1])])
    if any(name in file_name for name in in_list):
        scale /= 2.54
    elif any(name in file_name for name in cm_list):
        pass
    else:
        scale /= 2.54
        print("Unable to determine unit. Defaulting to cm.")
    return scale


def check(arr, val, flipped):
    if flipped:
        return arr > val
    return arr < val


def gen_mask(bbox, file_path, file_name, im_gray, val, detectron_mask, flipped=False):
    """
    Generates the mask for the fish and floodfills to make a whole image.
    """
    failed = False
    left = round(bbox[0])
    right = round(bbox[2])
    top = round(bbox[1])
    bottom = round(bbox[3])
    bbox_orig = bbox
    bbox = (left, top, right, bottom)

    im = im_gray.copy()
    shape = im.shape
    done = False
    im_crop = im[top:bottom, left:right]
    fish_pix, thresh, new_mask = None, None, None

    while not done:
        done = True
        im_crop = im[top:bottom, left:right]
        count = 0
        thresh = np.where(im_crop < val, 1, 0).astype(np.uint8)
        indices = list(zip(*np.where(thresh == 1)))
        shuffle(indices)
        for ind in indices:
            if fish_pix is not None:
                ind = fish_pix
            count += 1
            # if 10k pass and fish not found
            if count > 10000:
                if fish_pix is not None:
                    fish_pix = None
                else:
                    print(f'ERROR on flood fill: {file_name}')
                    return bbox_orig, detectron_mask.astype('uint8'), True
            temp = flood_fill(thresh, ind, 2)
            temp = np.where(temp == 2, 1, 0)
            percent = np.count_nonzero(temp) / im_crop.size
            if percent > 0.1:
                fish_pix = ind
                for i in (0, temp.shape[0] - 1):
                    for j in (0, temp.shape[1] - 1):
                        temp = flood_fill(temp, (i, j), 2)
                thresh = np.where(temp != 2, 1, 0).astype(np.uint8)
                break
        new_mask = np.full(shape, 0).astype(np.uint8)
        new_mask[top:bottom, left:right] = thresh
        # Expands the bounding box
        try:
            if np.any(new_mask[top:bottom, left] != 0) and left > 0:
                left -= 1
                left = max(0, left)
                done = False
            if np.any(new_mask[top:bottom, right] != 0) and right < shape[1] - 1:
                right += 1
                right = min(shape[1] - 1, right)
                done = False
            if np.any(new_mask[top, left:right] != 0) and top > 0:
                top -= 1
                top = max(0, top)
                done = False
            if np.any(new_mask[bottom, left:right] != 0) and bottom < shape[0] - 1:
                bottom += 1
                bottom = min(shape[0] - 1, bottom)
                done = False
        except:
            print(f'{file_name}: Error expanding bounding box')
            # done = True
            return bbox_orig, detectron_mask.astype('uint8'), True
        # New bbox
        bbox = (left, top, right, bottom)
        # New threshold
        val = adaptive_threshold(bbox, im_gray)
    if np.count_nonzero(thresh) / im_crop.size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        new_mask = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    return bbox, new_mask, failed


def gen_mask_upscale(bbox, file_path, file_name, im_gray, val, detectron_mask):
    failed = False
    l = round(bbox[0])
    r = round(bbox[2])
    t = round(bbox[1])
    b = round(bbox[3])
    bbox_orig = bbox
    bbox = (l, t, r, b)

    im = im_gray.copy()
    im_crop = im[t:b, l:r]
    thresh = np.where(im_crop < val, 1, 0).astype(np.uint8)
    new_mask = np.full(im.shape, 0).astype(np.uint8)
    new_mask[t:b, l:r] = thresh
    if np.count_nonzero(thresh) / im_crop.size < .1:
        print(f'{file_name}: Using detectron mask and bbox')
        new_mask = detectron_mask.astype('uint8')
        bbox = bbox_orig
        failed = True
    return bbox, new_mask, failed


def gen_metadata_safe(file_path):
    """
    Deals with erroneous metadata generation errors.
    """
    try:
        result, mask_uint8 = gen_metadata(file_path)
        return result, mask_uint8
    except Exception as e:
        print(f'{file_path}: Errored out ({e})')
        return {file_path: {'errored': True}}


def main(input_file, output_result, output_mask=None):

    result, mask_uint8 = gen_metadata_safe(input_file)

    with open(output_result, 'w') as f:
            json.dump(result, f)

    if output_mask != None:
        cv2.imwrite(output_mask, mask_uint8)

    return print('This is the user version 2.1!')

if __name__ == '__main__':

    if len(sys.argv) > 2: # case with 2 arguments input
        input_file = sys.argv[1]
        output_json = sys.argv[2]
        output_mask = None

    if len(sys.argv) == 4 : # case if there is a 3rd argument input
        output_mask = sys.argv[3]

    main(input_file, output_json, output_mask)
