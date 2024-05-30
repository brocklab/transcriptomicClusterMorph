# %%
from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

# %%
increasingSubpop = imread('../../figures/publication/results/increasingBBDemonstration.png')

subPopResults = imread('../../figures/publication/results/increasingBBSubpop.png')

fig, axs = plt.subplots(2, figsize = (4, 4), height_ratios = [1.75,3], layout = 'constrained')
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

axs[0].imshow(increasingSubpop, aspect = 'auto')
axs[0].axis('off')
axs[0].text(0.0, 1.0, 'a)', transform=axs[0].transAxes + trans,
        fontsize='large', weight = 'bold', va='top', ha = 'left')
axs[1].imshow(subPopResults, aspect = 'auto')
axs[1].axis('off')
axs[1].text(0.0, 1.0, 'b)', transform=axs[1].transAxes + trans,
        fontsize='large', weight = 'bold', va='top', ha = 'left')

fig.savefig('../../figures/publication/multipanel/increasingBB.png', 
            dpi = 500,
            bbox_inches = 'tight')
# %%
increasingSubpop436 = imread('../../figures/publication/results/increasingBBDemonstration436.png')

subPopResults436 = imread('../../figures/publication/results/increasingBBSubpop436.png')

fig, axs = plt.subplots(2, figsize = (4, 4), height_ratios = [1.75,3], layout = 'constrained')
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)

axs[0].imshow(increasingSubpop436, aspect = 'auto')
axs[0].axis('off')
axs[0].text(0.0, 1.0, 'a)', transform=axs[0].transAxes + trans,
        fontsize='large', weight = 'bold', va='top', ha = 'left')
axs[1].imshow(subPopResults436, aspect = 'auto')
axs[1].axis('off')
axs[1].text(0.0, 1.0, 'b)', transform=axs[1].transAxes + trans,
        fontsize='large', weight = 'bold', va='top', ha = 'left')

fig.savefig('../../figures/publication/multipanel/increasingBB436.png', 
            dpi = 500,
            bbox_inches = 'tight')

# %%
augmentations = imread('../../figures/publication/results/augmentationsDemonstration.png')
augmentationResults = imread('../../figures/publication/results/subPopulationAugmentation.png')
# %%

fig, axs = plt.subplots(2, figsize = (4, 5.5), height_ratios = [1.25,3], layout = 'constrained')

axs[0].imshow(augmentations, aspect = 'auto')
axs[0].axis('off')
trans = mtransforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
axs[0].text(0.0, 1.0, 'a)', transform=axs[0].transAxes + trans,
        fontsize='large', va='top', ha = 'left', weight = 'bold')
axs[1].imshow(augmentationResults, aspect = 'auto')
axs[1].axis('off')
axs[1].text(0.0, 1.0, 'b)', transform=axs[1].transAxes + trans,
        fontsize='large', weight = 'bold', va='top', ha = 'left')
fig.savefig('../../figures/publication/multipanel/augmentations.png', dpi = 500)
# %%