from pycocotools.coco import COCO as BaseCOCO
import json

class COCO(BaseCOCO):
    def __init__(self, annotation_file=None):
        if annotation_file is not None:
            with open(annotation_file, encoding="utf-8") as f:
                dataset = json.load(f)
        else:
            dataset = {}
        super().__init__()
        self.dataset = dataset
        self.createIndex()