import json
import random
import os
import sys
import yaml
import pathlib
import numpy as np
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data import Metadata
from detectron2.data import DatasetCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode


enhance = json.load(open('config/config.json', 'r'))
ENHANCE = bool(enhance['ENHANCE'])

with open('config/mask_rcnn_R_50_FPN_3x.yaml', 'r') as f:
    iters = yaml.load(f, Loader=yaml.FullLoader)["SOLVER"]["MAX_ITER"]


def import_config(config_data):
    
    
    config_file = json.load(open(config_data))
    
    #ENHANCE = config_file['ENHANCE'] 
    annotation_prefix = config_file['annotation_prefix']
    dataset_json = config_file['dataset_json']
    data_prefix = config_file['data_prefix']
    labels_file = config_file['labels']
    labels_f = open(labels_file,'r')
    data = labels_f.read()
    labels_list = [l for l in data.split("\n") if l != '' ]
    
    return annotation_prefix, dataset_json, data_prefix, labels_list # , ENHANCE

def register_data(config_data):
        
    annotation_prefix,\
        dataset_json, prefix,\
            labels_list = import_config(config_data)
    
    mapping = {i+1:i for i in range(len(labels_list))}
    conf = json.load(open(dataset_json))
    metadata = None  # Need it in outer block for reuse
    train = []

    for img_dir in conf.keys():
        #ims = f'{prefix}{img_dir}'
        ims = os.path.join(prefix, img_dir)
        for dataset in conf[img_dir]:
            json_file = os.path.join(annotation_prefix, dataset)
            #json_file = f'datasets/{dataset}'
            name = dataset.split('.')[0]
            train.append(name)
            register_coco_instances(name, {}, json_file, ims)
            
    return train
        
def main(config_data, enhance_contrast=ENHANCE):

    #train = import_config(config_data)
    train = register_data(config_data)
    cfg = get_cfg()
    #cfg.OUTPUT_DIR += "/enhance"
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = tuple(train)
    cfg.DATASETS.TEST = ()  # no metrics implemented yet
    cfg.DATALOADER.NUM_WORKERS = 2
    # initialize from model zoo
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.02
    # cfg.SOLVER.MAX_ITER = (50000)

    ################
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    ################

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128)

    #cfg.OUTPUT_DIR += f"/non_enhanced_{iters}" if not enhance_contrast else f"/enhanced_{iters}"
    print(f"{cfg.OUTPUT_DIR}")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=True)
    trainer.train()

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    predictor = DefaultPredictor(cfg)

    return print(f'Training over, model saved in {cfg.OUTPUT_DIR}')


if __name__ == '__main__':
    
    main(sys.argv[1])
