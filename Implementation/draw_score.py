import pandas as pd
import matplotlib.pyplot as plt

file_name = 'tdGammonR'
fig = pd.read_pickle(file_name+'.gz', compression = 'gzip')

fig = fig.groupby(fig.index//20).mean()
#fig = fig.interpolate(method = 'cubic')

fig.plot()
plt.suptitle(file_name)
win = plt.gcf()
win.canvas.set_window_title(file_name)
plt.show()