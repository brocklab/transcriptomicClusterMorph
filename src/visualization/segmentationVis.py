from src.data.fileManagement import getImageBase

import matplotlib.pyplot as plt

from skimage.io import imread
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode


def viewPredictorResult(predictor, imPath: str):
    """
    Plots an image of cells with masks overlaid.
    Inputs:
    predictor: A predictor trained with detectron2
    imPath: The path of the image to load
    Outputs:
    None
    """
    im = imread(imPath)
    imBase = getImageBase(imPath.split('/')[-1])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                #    metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure()
    print("plotting")
    plt.imshow(out.get_image()[:,:,::-1])
    plt.title(imBase)
    plt.show()
