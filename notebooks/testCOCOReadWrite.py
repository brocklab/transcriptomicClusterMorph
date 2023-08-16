# %%
import detectron2.data.datasets as datasets
import json

from detectron2.data import MetadataCatalog, DatasetCatalog
# %%
with open('./TJ2201Segmentations.json') as f:
    d = json.load(f)
# %%
datasetDicts = datasets.load_coco_json(json_file='./TJ2201Segmentations.json', image_root='')
# %%
def getCells(datasetDict):
    return datasetDict

inputs = [datasetDicts]
if 'cellMorph' in DatasetCatalog:
    DatasetCatalog.remove('cellMorph')
    MetadataCatalog.remove('cellMorph')

DatasetCatalog.register("cellMorph", lambda x=inputs: getCells(inputs[0]))
MetadataCatalog.get("cellMorph").set(thing_classes=["cell"])

datasets.convert_to_coco_json('cellMorph', output_file='./TJ2201Segmentations2.json')
