import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.stats import percentileofscore

use_precomputed = True

if use_precomputed == True:
    scores = pd.read_csv(f'data/spis.csv',index_col=0)
    nulls = pd.read_csv(f'data/null.csv',index_col=0)
    score_all = pd.read_csv(f'data/all.csv',index_col=0)
else:
    scores = pd.read_csv(f'results/BasicMotions-spis.csv',index_col=0)
    nulls = pd.read_csv(f'results/BasicMotions-spis-null.csv',index_col=0)
    score_all = pd.read_csv(f'results/BasicMotions-score.csv',index_col=0)

categories = pd.read_csv('data/categories.csv',index_col=0)
modules = pd.read_csv('data/modules.csv',index_col=0)

categories = categories.rename(columns={'0':'class'})
modules = modules.rename(columns={'0':'module'})

dorder = ['causal','distance','infotheory','spectral','misc','basic','wavelet']
morder = modules['module'].unique()
morder.sort()

dcmap = sns.color_palette('pastel',len(dorder))
mcmap = [mpl.colors.rgb2hex(c/255+(0,)) for c in np.genfromtxt('data/cmap.csv', delimiter=',')]

nulls['average'] = nulls[[col for col in scores.columns if 'resample' in col]].mean(axis=1)
scores['average'] = scores[[col for col in scores.columns if 'resample' in col]].mean(axis=1)

# Remove any that completely failed
nulls = nulls[~scores['average'].isna()]
scores = scores[~scores['average'].isna()]

avgn = nulls['average']
avga = scores['average']
print(f'Bottom scores (actuals): {avga.sort_values()[:10]}')
print(f'Top scores (actuals): {avga.sort_values()[-10:]}')

vfunc = np.vectorize(lambda x : percentileofscore(nulls['average'][~nulls['average'].isna()],x))

p = 1 - vfunc(scores['average'].values) / 100
pvalues = pd.DataFrame(columns=['pvalue'],index=scores.index,data=p)

bf_alpha = 0.05 / scores.shape[0]
scores = pd.concat([scores,pd.DataFrame(categories),pd.DataFrame(modules)],axis=1,join="inner")

all_score = score_all.mean().values[0]
null_quant = np.quantile(nulls['average'][~nulls['average'].isna()],1-bf_alpha)
scores['significant'] = pvalues < bf_alpha

print('Variance by category: {}'.format(scores[['category','average']].groupby('category').var()))
print('Variance by category: {}'.format(scores[['module','average']].groupby('module').var()))

# Plot the histogram of scores coloured by significance (from Fisher's method)
fig, ax = plt.subplots(figsize=(5,3))
sns.histplot(data=scores,hue='significant',x='average',multiple='stack',bins=20,element='step',stat='probability',palette='binary')
plt.axvline(null_quant,color='k')
plt.axvline(all_score,color='r')

# Plot the violin plot of the average score by category
fig, ax = plt.subplots()
sortorder = pd.Series(['basic','distance','causal','infotheory','spectral','misc'])
cmap = np.array(dcmap)[sortorder.map({k: i for i, k in enumerate(dorder)})]
sns.violinplot(data=scores,x='category',y='average',palette=cmap,cut=0,order=sortorder,scale='width',inner=None)
sns.stripplot(data=scores,x='category',y='average',jitter=True,color='#000000',order=sortorder,alpha=0.6)
plt.axhline(null_quant,color='k')
plt.axhline(all_score,color='r')

# Plot the violin plot of the average score by module
fig, ax = plt.subplots(figsize=(7.5,5))
sortorder = scores[['module','average']].groupby(by='module').mean().sort_values(by='average').index
sns.violinplot(data=scores,x='module',y='average',palette=mcmap,cut=0,order=morder,scale='width',width=0.6,linewidth=1,inner=None)
sns.stripplot(data=scores,x='module',y='average',jitter=True,color='#000000',order=morder,alpha=0.6)
plt.axhline(null_quant,color='k')
plt.axhline(all_score,color='r')
plt.show()