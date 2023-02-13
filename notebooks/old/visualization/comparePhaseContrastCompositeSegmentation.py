# %%
import src.data.cellMorphHelper as cellMorphHelper

import pickle
import random
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
esamPos = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
cells = esamNeg+esamPos
random.shuffle(cells)
# %%
predictorPhaseContrast = cellMorphHelper.getSegmentModel('../models/AG2021Split16')
predictorComposite = cellMorphHelper.getSegmentModel('../models/AG2021Split16Composite')

# %%
num = random.randint(0, len(cells))
phaseContrast = cells[num].phaseContrast
composite =     cells[num].composite
 
cellMorphHelper.viewPredictorResult(predictorPhaseContrast, phaseContrast)
cellMorphHelper.viewPredictorResult(predictorComposite, composite)
