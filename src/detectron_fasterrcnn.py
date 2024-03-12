from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
import os
import torch

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# Link to the model weights
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 500 

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

# Freeze the backbone layers
cfg.MODEL.BACKBONE.FREEZE_AT = 2

class Trainer(DefaultTrainer):
    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Custom function to build the optimizer, allowing us to exclude
        certain parameters from the optimizer (thus freezing them)
        """
        frozen_parameters = set()
        for name, parameter in model.named_parameters():
            if "backbone" in name:  # Freeze layers by adjusting this condition
                frozen_parameters.add(name)
                parameter.requires_grad_(False)

        params = []
        for key, value in model.named_parameters():
            if key not in frozen_parameters and value.requires_grad:
                params.append({"params": [value]})
        optimizer = torch.optim.SGD(params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        return optimizer

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
