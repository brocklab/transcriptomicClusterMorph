# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
lpdSample = pd.read_csv('../../../data/misc/lpdSample-new.csv',
                        header=None).T
lpdSample = lpdSample.iloc[1:]
lpdSample.set_index(0, inplace=True)
lpdSample.index.name = None
lpdSample.columns = ['well', 'proportion']
lpdSample['proportion'] = lpdSample['proportion'].astype('float')
lpdSample['mode'] = 'imaging'

lpdSample.head()
# %%
prop1 = 0.161589*100
lpdSample.loc[lpdSample.shape[0]+1] = ['NA', prop1, 'sequenced']

# %%
lpdImaging = lpdSample.loc[lpdSample['mode'] == 'imaging']

plt.hist(lpdImaging['proportion'], bins = 'auto')
plt.axvline(prop1, color = 'red')
# %%
