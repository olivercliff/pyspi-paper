import random

import matplotlib.pyplot as plt, matplotlib as mpl
import pandas as pd, numpy as np
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram, set_link_color_palette

from utils import draw_network

random.seed(1)

mm_adj = pd.read_csv('data/mm_adj.csv',header=0,index_col=0)
statnames = mm_adj.columns

np.fill_diagonal(mm_adj.values,np.nan)

method = 'weighted'

y = 1 - mm_adj.fillna(0).values[np.triu_indices(mm_adj.shape[0],1)]
Z = linkage(y,metric='euclidean',method=method,optimal_ordering=True)


threshold = 0.76
clusters = fcluster(Z,threshold,criterion='distance')
nclusters = np.unique(clusters).size

cm = np.genfromtxt('data/cmap.csv', delimiter=',')
cmap = [mpl.colors.rgb2hex(c/255+(0,)) for c in cm]

set_link_color_palette(cmap)
fig, ax = plt.subplots(figsize=(20, 5))
dn = dendrogram(Z,labels=statnames,orientation='top',
                    color_threshold=threshold,leaf_font_size=4,
                    count_sort='ascending',above_threshold_color='k')
plt.axvline(x=threshold, c='grey', lw=1, linestyle='dashed')

colours = {s : mpl.colors.hex2color(c) for s,c in zip(dn['ivl'],dn['leaves_color_list'])}
_, uidx = np.unique(dn['leaves_color_list'], return_index=True)
unique_modules = np.array(dn['leaves_color_list'])[np.sort(uidx)]
lut = {f'M{i+1:02d}' : mpl.colors.hex2color(c) for i, c in enumerate(unique_modules)}

lut['Singleton'] = (0,0,0)
del lut[[k for k, v in zip(lut.keys(),lut.values()) if v == (0,0,0)][0]]

lut_r = {m: c for c, m in zip(lut.keys(),lut.values())}

modules = pd.Series(colours).map(lut_r)

modules.iloc[13:][modules.iloc[13:] == 'M01'] = 'M14'
modules.loc[dn['ivl']].to_csv('data/modules.csv')

focal_stats = [
                ('dcorr',0.5),
                ('lcss',0.75),
                ('gc_gaussian_k-max-10_tau-max-2',0.7),
                ]

for f, cutoff in focal_stats:
    cmodules = fcluster(Z,cutoff,criterion='distance')
    statmodmap = {f: m for f,m in zip(statnames,cmodules)}
    try:
        mod = statmodmap[f]
        mystats = [s for s in statnames if statmodmap[s] == mod]
    except KeyError:
        print(f'Statistic {f} not in matrix.')
        continue
    myadj = mm_adj.loc[mystats,mystats]

    ts = (0.75,0.5,0.25)
    draw_network(myadj,f=f,squared=True,node_color=colours,color_labels=lut,seed=1,pos=None,labels_on=True,ts=ts,ws=(2,0.2,0.05))

plt.show()