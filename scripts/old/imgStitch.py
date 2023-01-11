# %% [markdown]
"""
# Stitching masks back together
"""
# %%
import os
import cv2
import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel
# %%
def newGrid(n: int) -> list:
    """
    Makes grid outputs align with matplotlib subplots. IE grid numbering for a 3x3 split looks like:

    1, 4, 7
    2, 5, 8
    3, 6, 9

    However subplots do not go in this order. This outputs a list that allows for loading of the correct images
    """
    newGrid = []
    for i in range(1, n+1):
        gridNum = i
        for j in range(1, n+1):
            newGrid.append(gridNum)
            gridNum+=n
    return newGrid
# %%
predictor = getSegmentModel('../models/AG2021Split16')
# %%

# Get proper grid
imPath = '../data/AG2021Split16/phaseContrast'
testIm = 'phaseContrast_C5_1_2020y06m19d_00h33m'

fullImage = cv2.imread(os.path.join('../data/AG2021/phaseContrast', testIm+'.jpg'))

imNum = 1
plt.figure(figsize=(5,5))
imCt = 1
for imNum in newGrid(4):
    imPathFull = os.path.join(imPath, testIm+'_'+str(imNum)+'.jpg')

    im = cv2.imread(imPathFull)

    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],
                    metadata=cell_metadata, 
                    scale=1, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.subplot(4,4,imCt)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.axis('off')
    plt.title(imNum)
    imCt += 1
plt.show()
# %%
imNum = 9
imPathFull = os.path.join(imPath, testIm+'_'+str(imNum)+'.jpg')

im = cv2.imread(imPathFull)

outputs = predictor(im)['instances']

nCells = len(outputs)

if nCells>0:
    allMasks = np.zeros(outputs.image_size)
    for nCell in range(nCells):
        mask = outputs[nCell].pred_masks.numpy()[0]
        mask = clear_border(mask)
        allMasks = allMasks+mask
# %%


print(newGrid)
