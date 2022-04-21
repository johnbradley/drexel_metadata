import json
import random
import os
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


def visualize_input(metadata, count):
    name = metadata.get("name")
    dataset_dicts = DatasetCatalog.get(name)
    for d in random.sample(dataset_dicts, count):
        full_path = d['file_name']
        file_name = d['file_name'].split('/')[-1]
        img = cv2.imread(full_path)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        vis = visualizer.draw_dataset_dict(d)
        dirname = 'images/'
        dirname += 'enhanced/' if ENHANCE else 'non_enhanced/'
        os.makedirs('images', exist_ok=True)
        os.makedirs('images/enhanced', exist_ok=True)
        os.makedirs('images/non_enhanced', exist_ok=True)
        print(f'{dirname}/{name}_{file_name}')
        cv2.imwrite(f'images/{name}_{file_name}', vis.get_image()[:, :, ::-1])


def visualize_output(image_path, im_nb, predictor, metadata, enhance_contrast):
    
    #i = 0
    #names = os.listdir(test_images)
    path =pathlib.Path(image_path)
    name = path.stem
    
    im = cv2.imread(image_path)
    outputs = []
    # apply contrast enhancement technique
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
        
    output = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=metadata,
                    scale=0.8,
                    # remove the colors of unsegmented pixels
                    instance_mode=ColorMode.IMAGE_BW
                    )
    v = v.draw_instance_predictions(output[-1]["instances"].to("cpu"))
    
    dirname = 'images/'
    dirname += 'enhanced' if enhance_contrast else 'non_enhanced'
    os.makedirs('images', exist_ok=True)
    os.makedirs('images/enhanced', exist_ok=True)
    os.makedirs('images/non_enhanced', exist_ok=True)
    print(f'{dirname}/prediction_{name}')
    cv2.imwrite(f'{dirname}/prediction_{name}', v.get_image()[:, :, ::-1])


def main(enhance_contrast=ENHANCE):
    print(f"{ENHANCE=} {iters=}")
    
    prefix = open('config/overall_prefix.txt').readlines()[0].strip()
    conf = json.load(open('config/training_data_all.json'))
    metadata = None  # Need it in outer block for reuse
    train = []
    test_images = f'{prefix}tulane/'

    for img_dir in conf.keys():
        ims = f'{prefix}{img_dir}'
        for dataset in conf[img_dir]:
            json_file = f'datasets/{dataset}'
            name = dataset.split('.')[0]
            train.append(name)
            # This if only matters if you want to visualize a certain
            # dataset with the `visualize_input` function after the loop.
            # Otherwise, any of the datasets will work.
            if name == '1':
                metadata = Metadata(evaluator_type='coco', image_root=ims,
                                    json_file=json_file,
                                    name=name,
                                    thing_classes=[
                                        'fish', 'ruler', 'eye', 'two', 'three'],
                                    thing_dataset_id_to_contiguous_id={
                                        1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
                                    )
            register_coco_instances(name, {}, json_file, ims)


    cfg = get_cfg()
    #cfg.OUTPUT_DIR += "/enhance"
    cfg.merge_from_file("config/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = tuple(train)
    cfg.DATASETS.TEST = ()  # no metrics implemented yet
    cfg.DATALOADER.NUM_WORKERS = 2
    # initialize from model zoo
    cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.02
    # cfg.SOLVER.MAX_ITER = (
    # 50000
    # )

    ################
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    ################

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        128
    )

    cfg.OUTPUT_DIR += f"/non_enhanced_{iters}" if not enhance_contrast else f"/enhanced_{iters}"
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
    main()
