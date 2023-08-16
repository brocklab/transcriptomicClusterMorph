# %%
import matplotlib.pyplot as plt
import re
# %%
with open('./aaeRes.txt') as outFile:
    x = outFile.read()
x = x.split('\n')

GLoss, DLoss = [], []
for line in x:
    losses = re.findall("\d+\.\d+", line)
    if len(losses) == 2:
        DLoss.append(float(losses[0]))
        GLoss.append(float(losses[1]))

# %%
plt.subplot(121)
plt.plot(DLoss)
plt.subplot(122)
plt.plot(GLoss)
# %%
