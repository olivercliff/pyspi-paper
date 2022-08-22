import numpy as np
import pandas as pd
import os

# mpl.use('GTK3Agg') # uncomment if you're having problems with multithreading (but you'll need cairo)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec

import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

import sklearn.cluster as cluster
from scipy.stats import zscore
import networkx as nx

def _reweight(x,ts=(0.75,0.5,0.25),ws=(2,0.75,0.1)):
    for t, w in zip(ts,ws):
        if x >= t:
            return w
    return 0

def _nudge(pos, x_shift, y_shift):
    return {n:(x + x_shift, y + y_shift) for n,(x,y) in pos.items()}

def draw_network(adj,f=None,squared=False,node_color=None,color_labels=None,labels_on=False,pos=None,layout='spring',seed=1,use_kk=True,savedir=None,ts=None,ws=None,alpha=None):

    if ts is None:
        if adj.shape[0] < 50:
            ts = (0.9,0.7,0.5)
        else:
            vec = adj.values[np.triu_indices(adj.shape[0],1)]
            ts = np.percentile(vec[~np.isnan(vec)],[99,95,90])
    if ws is None:
        ws = (2,0.75,0.1)
    if squared:
        adj = adj**2
        ts = [t**2 for t in ts]

    if alpha is None:
        alpha = 1

    G = nx.from_pandas_adjacency(adj)
    if f is not None:

        fig, ax = plt.subplots(1,figsize=(10,7))

        if layout == 'spring':
            if pos is None:
                if use_kk:
                    pos = nx.kamada_kawai_layout(G)
                else:
                    pos = None
            pos = nx.spring_layout(G,pos=pos,seed=seed,iterations=1000)
        elif pos is None:
            raise ValueError('pos must be included if layout is not spring.')

        weights = [_reweight(G[u][v]['weight'],ts=ts,ws=ws) for u, v in G.edges()]
        nx.draw(G,pos=pos,ax=ax,with_labels=False,node_size=250,
                    edgecolors='k',edge_color=None,width=weights,
                    node_color=[node_color[f] for f in pos],alpha=alpha)

        pos_labels = _nudge(pos,0,0.02)
        if labels_on:
            nx.draw_networkx_labels(G,pos=pos_labels,ax=ax,font_size=6)
            plt.margins(x=0.4)
        plt.tight_layout()
        title = f'network-{f}'
    else:
        fig, ax = plt.subplots(1,figsize=(10,10))
        weights = [_reweight(G[u][v]['weight'],ts=ts,ws=ws) for u, v in G.edges()]

        if layout == 'spring':
            if use_kk:
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = None
            pos = nx.spring_layout(G,seed=seed,pos=pos,iterations=1000)
        elif pos is None:
            raise ValueError('pos must be included if layout is not spring.')

        _ = nx.draw_networkx_nodes(G,pos=pos,ax=ax,node_size=150,
                                edgecolors=[[0.8*c for c in node_color[f]] for f in pos],
                                linewidths=1,
                                node_color=[node_color[f] for f in pos],alpha=alpha)
        _ = nx.draw_networkx_edges(G,pos=pos,ax=ax,edge_color=None,width=weights,alpha=alpha)

        if labels_on:
            _ = nx.draw_networkx_labels(G,pos=pos,ax=ax,font_size=1)
            plt.margins(x=0.4)
        title = 'network'
        
    ax = plt.gca()
    if color_labels is not None:
        ns = []
        for l in color_labels:
            ns.append(ax.scatter([],[],color=color_labels[l],label=l))

        lines = []
        for t, w in zip(ts,ws):
            if squared:
                l, = ax.plot([],[],color='k',linewidth=w,label=f'|r| > {np.sqrt(t):.2f}')
            else:
                l, = ax.plot([],[],color='k',linewidth=w,label=f'|r| > {t:.2f}')
            lines.append(l)
        legend1 = plt.legend(lines,[l.get_label() for l in lines],loc=3)
        ax.legend(ns,[n.get_label() for n in ns],loc=1)
        ax.add_artist(legend1)

    plt.tight_layout()
    plt.axis('off')
    if savedir is not None:
        path = os.path.join(savedir,title+'.pdf')
        fig.savefig(path,dpi=300)
        path = os.path.join(savedir,title+'.png')
        fig.savefig(path,dpi=300)
        print(f'Saving network to {path}.')
        plt.close(fig)

def _despine(ax):
    for side in ['left','right','top','bottom']:
        ax.spines[side].set_visible(False)

def plot_clusters(mm_adj,cols=None,col_labels=None,method='average',min_rho=None,apx='',mask_on=False,savedir=None):
    mask = mm_adj.isnull()
    mm_adj[mask] = 0

    if min_rho is not None:
        vmin = min_rho
    else:
        vmin = np.min(mm_adj.values)

    y = 1-mm_adj.abs().values[np.triu_indices(mm_adj.shape[0],1)]
    Z = linkage(y,metric='euclidean',method=method,optimal_ordering=True)

    fig, ax = plt.subplots(figsize=(3, 16))
    dn = dendrogram(Z,labels=mm_adj.columns.values,orientation='left',
                        color_threshold=0,count_sort='ascending',above_threshold_color='k')
    plt.axis('off')

    if savedir is not None:
        fig.savefig(savedir+'/dendrogram-sm' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)

    # The average (spearman) correlation between measures
    mm_adj[mask] = np.NaN
    mm_adj_sort = mm_adj.loc[dn['ivl'],dn['ivl']]

    if mask_on:
        mm_adj_sort.values[np.triu_indices(mm_adj_sort.shape[0], 1)] = np.nan

    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.pcolormesh(mm_adj_sort.values,vmin=vmin,vmax=1,
                        cmap=plt.cm.get_cmap('RdYlBu_r',9))
    plt.tick_params(which='both', bottom=False,top=False,left=False,right=False,
                    labelbottom=False,labelleft=False)
    ax.invert_yaxis()
    plt.axis('off')
    if savedir is not None:
        fig.savefig(savedir+'/mm_cluster' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(1,8))
    fig.colorbar(im,cax=ax)
    ax.tick_params(labelsize=30)
    if savedir is not None:
        fig.savefig(savedir+'/colorbar' + apx + '.jpg',dpi=300,bbox_inches='tight',pad_inches=0)
        plt.close(fig)

    if cols is not None:
        row_cols = np.array([cols[f] for f in dn['ivl']])
        fig, ax = plt.subplots(figsize=(12,2))
        im = ax.imshow(np.repeat(row_cols.reshape((1,len(cols),3)),15,axis=0))
        plt.axis('off')
        if savedir is not None:
            fig.savefig(savedir+'/colorrow' + apx + '.pdf',dpi=300,bbox_inches='tight',pad_inches=0)
            plt.close(fig)

def animate_network(adj0,adj1,node_color,name=None,savefile=None):

    fig, ax = plt.subplots(figsize=(10,10))

    G0 = nx.from_pandas_adjacency(adj0)
    G1 = nx.from_pandas_adjacency(adj1)

    remove = [node for node, degree in dict(G0.degree(weight='weight')).items() if degree < 5]
    G0.remove_nodes_from(remove)
    G1.remove_nodes_from(remove)

    pos0 = nx.spring_layout(G0)
    pos1 = nx.spring_layout(G1,pos=pos0)

    nodes = nx.draw_networkx_nodes(G0,pos=pos0,ax=ax,node_size=150,
                                    edgecolors=None,linewidths=None,
                                    node_color=[node_color[f] for f in pos0],alpha=0.75)

    # weights = [_reweight(G0[u][v]['weight']) for u, v in G0.edges()]
    # edges = nx.draw_networkx_edges(G0,pos=pos0,ax=ax,edge_color=None,width=weights,alpha=0.3)

    nx.draw_networkx_nodes(G1,pos=pos1,ax=ax,node_size=150,
                                edgecolors=None,linewidths=None,
                                node_color=[node_color[f] for f in pos0],alpha=0.1)
    plt.axis('off')

    frames = 200

    x0 = [pos0[f][0] for f in pos0]
    y0 = [pos0[f][1] for f in pos0]
    x1 = [pos1[f][0] for f in pos0]
    y1 = [pos1[f][1] for f in pos0]

    xs = np.linspace(x0,x1,frames)
    ys = np.linspace(y0,y1,frames)

    def animate(i):
        npos = np.array([xs[i,:],ys[i,:]]).T
        nodes.set_offsets(npos)
        # edges.set_offsets(npos)
        return nodes,

    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    if name is not None:
        plt.suptitle(name)
    if savefile is not None:
        print(f'Saving animation to {savefile}.')
        anim.save(savefile)
        plt.close(fig)

def asframe(func):
    def convert(calcs,**kwargs):
        if isinstance(calcs,Calculator):
            cf = CalculatorFrame()
            cf.add_calculator(calcs)
            return func(cf,**kwargs)
        if isinstance(calcs,list) and isinstance(calcs[0],Calculator):
            cf = CalculatorFrame(calculators=calcs)
            return func(cf,**kwargs)
        elif isinstance(calcs,CalculatorFrame):
            return func(calcs,**kwargs)
        else:
            raise TypeError('First parameter must be either a list of Calculators or a CalculatorFrame.')

    return convert

def diagnostics(calc):
    """ TODO: print out all diagnostics, e.g., compute time, failures, etc.
    """
    sid = np.argsort(calc._proctimes)
    print(f'Processing times for all {len(sid)} measures:')
    for i in sid:
        print('[{}] {}: {} s'.format(i,calc._measure_names[i],calc._proctimes[i]))

def rasterplot(data,cmap='icefire',window=7,proc_cluster=True,animate=True,savefilename=None):

    if isinstance(data,np.ndarray):
        data = Data(data)
    dat = data.to_numpy(squeeze=True)

    if animate:
        figsize=(10,10)
        dendrogram_ratio = 0.1
        cbar_pos=None
    else:
        figsize=(7,10)
        cbar_pos=(0, .2, .03, .4)
        dendrogram_ratio = 0.1

    g = sns.clustermap(np.transpose(dat),
                            cmap=cmap,figsize=figsize,
                            col_cluster=proc_cluster,row_cluster=False,
                            dendrogram_ratio=dendrogram_ratio,cbar_pos=cbar_pos,
                            robust=True)
    ax_im = g.ax_heatmap
    fig = ax_im.figure

    ax_im.set_xlabel('Process')
    ax_im.set_ylabel('Time')
    ax_im.figure.suptitle(f'Space-time amplitude plot for "{data.name}"')

    if animate:
        g.gs.update(left=0.05, right=0.45, bottom=0.1, top=0.9)
        gs2 = gridspec.GridSpec(1,1, left=0.6, right=0.9, bottom=0.3, top=0.55)
        ax_st = g.fig.add_subplot(gs2[0,0])

        cols = sns.color_palette('Blues',n_colors=window)
        lines = []
        for t in range(window):
            lines.append(ax_st.plot(dat[:,t],color=cols[t])[0])

        def update_plots(ti,data,lines,ax):
            maxT = data.shape[1]
            for t, line in enumerate(lines):
                line.set_ydata(data[:,(ti+t)%maxT])
            ax.set_title(f'Amplitude at time t={ti}')

        lims = [np.min(dat),np.max(dat)]
        padding = np.ptp(lims)*0.05
        ax_st.set_ylim([lims[0]-padding,lims[1]+padding])
        ax_st.set_title('Time t=0')
        ax_st.set_xlabel('Process')
        ax_st.set_ylabel('Amplitude')

        repeat = True
        if savefilename is not None:
            repeat = False
        line_ani = animation.FuncAnimation(fig,update_plots,data.n_observations,
                                            fargs=(dat,lines,ax_st),interval=100,blit=False,repeat=repeat)

        ax_im.locator_params(axis='y', nbins=6)
        if savefilename is not None:
            fname = savefilename+'.gif'
            line_ani.save(fname, writer='imagemagick', fps=10)
            print(f'Saved gif to {fname}')
            plt.close(fig)
        else:
            plt.show()
    else:
        if savefilename is not None:
            fname = savefilename+'.jpg'
            fig.savefig(fname,format='jpg',bbox_inches='tight')
            print(f'Saved figure to {fname}')
            plt.close(fig)
        else:
            plt.show()

def mm_cluster(cf,dropna=True,absolute=False,classes=None,flatten_kwargs={},
                clustermap_kwargs={'cmap': plt.cm.get_cmap('RdYlBu_r',9), 'xticklabels': 1,'yticklabels': 1}):
    
    if not isinstance(cf,CorrelationFrame):
        cf = CorrelationFrame(cf)
    mdf = cf.mdf

    if absolute:
        mdf = mdf.abs()

    mm_adj = mdf.groupby(level='Source statistic').mean()
    mm_adj = mm_adj.sort_index().reindex(sorted(X),axis=1)
    
    if dropna:
        mm_adj = mm_adj.dropna(how='all',axis=0).dropna(how='all',axis=1)
        
    if absolute:
        mm_adj = mm_adj.abs()
        clustermap_kwargs['vmin'] = 0
        clustermap_kwargs['vmax'] = 1
    else:
        clustermap_kwargs['vmin'] = -1
        clustermap_kwargs['vmax'] = 1
    mm_adj.fillna(0,inplace=True)

    colors = None
    if classes is not None:
        cf.set_sgroups(classes)
        groups = pd.Series(cf.get_sgroup_names(mm_adj.columns))
        lut = dict(zip(groups.unique(),sns.color_palette('pastel', groups.unique().size)))
        colors = groups.map(lut).values

    if mm_adj.shape[0] > 20:
        sns.set(font_scale=0.5)
    g = sns.clustermap(mm_adj,col_colors=colors,row_colors=colors,**clustermap_kwargs)

    # Prettify
    ax = g.ax_heatmap
    ax_hmcb = g.ax_cbar
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    sns.set(font_scale=1)
    g.gs.update(top=0.9)
    g.fig.suptitle(f'Measure-measure clustermap for all {cf.ddf.shape[1]} datasets of "{cf.name}" frame')
    ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])
    
    return ax.figure

def dd_cluster(cf,absolute=True,flatten_kwargs={},
                clustermap_kwargs={'cmap':plt.cm.get_cmap('RdYlBu_r',9),'xticklabels': 1,'yticklabels' :1},
                classes=None):

    if not isinstance(cf,CorrelationFrame):
        cf = CorrelationFrame(cf)

    feature_matrix = cf.get_feature_matrix()
    dd_adj = feature_matrix.corr(method='spearman')
        
    mask = dd_adj.isna()
    dd_adj.fillna(0,inplace=True)
    if absolute:
        dd_adj = dd_adj.abs()
        clustermap_kwargs['vmin'] = 0
        clustermap_kwargs['vmax'] = 1
    else:
        clustermap_kwargs['vmin'] = -1
        clustermap_kwargs['vmax'] = 1

    colors = None
    if classes is not None:
        cf.set_dgroups(classes)
        groups = pd.Series(cf.get_dgroup_names(dd_adj.columns))
        lut = dict(zip(groups.unique(),sns.color_palette('pastel', groups.unique().size)))
        colors = groups.map(lut).values

    if dd_adj.shape[0] > 20:
        sns.set(font_scale=0.5)
    g = sns.clustermap(dd_adj,mask=mask,row_colors=colors,col_colors=colors,**clustermap_kwargs)

    # Prettify
    ax = g.ax_heatmap
    ax_hmcb = g.ax_cbar
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    sns.set(font_scale=1)
    g.gs.update(top=0.9)
    g.fig.suptitle(f'Data-data Clustermap for all {cf.ddf.shape[1]} datasets of "{cf.name}" frame')
    ax_hmcb.set_position([0.05, 0.8, 0.02, 0.1])
    
    return ax.figure

def _get_reducer(reducer):

    if reducer == 'pca':
        reducer = PCA(n_components=2, svd_solver='full')
        xlabel, ylabel = ('PC-1','PC-2')
    elif reducer == 'umap':
        from umap import UMAP
        reducer = UMAP()
        xlabel, ylabel = ('UMAP-1','UMAP-2')
    elif reducer == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2)
        xlabel, ylabel = ('tSNE-1','tSNE-2')
    return reducer, xlabel, ylabel


# Should really combine this and dataspace
def measurespace(cf,reducer='pca',classes=None,absolute=True,flatten_kwargs={}):

    if isinstance(cf,CalculatorFrame):
        cf = CorrelationFrame(cf,flatten_kwargs=flatten_kwargs)
    mdf = cf.mdf

    if absolute:
        mdf = mdf.abs()

    X = mdf.groupby('Source statistic').mean().fillna(0)
    X = X.sort_index().reindex(sorted(X),axis=1)
    
    if reducer != 'eig' and reducer != 'mds':
        reducer, xlabel, ylabel = _get_reducer(reducer)
        try:
            if absolute:
                embedding = reducer.fit_transform(np.abs(X))
            else:
                embedding = reducer.fit_transform(X)
        
            if isinstance(reducer,PCA):
                xlabel += f' ({100*reducer.explained_variance_ratio_[0]:.2f}%)'
                ylabel += f' ({100*reducer.explained_variance_ratio_[1]:.2f}%)'
        except ValueError as err:
            print(f'Dimensionality reduction failed: {err}.')
    elif reducer == 'mds':
        embedding = MDS(dissimilarity='precomputed').fit_transform(1-np.abs(X))
        # variance_explained = 1 - np.corrcoef(pdist(X),pdist(embedding))**2
        xlabel = 'MDS-1'
        ylabel = 'MDS-2'
    elif reducer == 'eig':
        B = zscore(X)
        B = np.nan_to_num(B)
        C = np.corrcoef(B)
        v, V = np.linalg.eig(C)
        order = np.argsort(v)[::-1]
        v = v[order]
        V = V[order]
        T = np.matmul(B,V)
        embedding = T[:,:2]
        xlabel = f'EV-1'
        ylabel = f'EV-2'

    embeddf = pd.DataFrame({xlabel: embedding[:,0], ylabel: embedding[:,1], 'measure': mdf.columns.tolist()},index=mdf.columns)
    
    if classes is not None:
        cf.set_sgroups(classes)
        embeddf['class'] = cf.get_sgroup_names(mdf.columns)

    fig, _ = plt.subplots(1,1)
    try:
        sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,hue='class',palette='pastel')
    except ValueError as err:
        sns.set(font_scale=0.5)
        sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,hue='measure',style='measure',palette='pastel')
        sns.set(font_scale=1.0)

    return fig, embeddf

def dataspace(cf,classes=None,reducer='pca',absolute=True,include_size=False,plot_nas=True,flatten_kwargs={},scatterplot_kwargs={}):
    if isinstance(cf,CalculatorFrame):
        cf = CorrelationFrame(cf,flatten_kwargs=flatten_kwargs)

    feature_matrix = cf.get_feature_matrix().fillna(0).T
    
    reducer, xlabel, ylabel = _get_reducer(reducer)
    try:
        if absolute:
            embedding = reducer.fit_transform(feature_matrix)
        else:
            embedding = reducer.fit_transform(feature_matrix.abs())

        if isinstance(reducer,PCA):
            xlabel += f' ({100*reducer.explained_variance_ratio_[0]:.2f}%)'
            ylabel += f' ({100*reducer.explained_variance_ratio_[1]:.2f}%)'

        embeddf = pd.DataFrame(index=feature_matrix.index,data=embedding,columns=[xlabel,ylabel])
    except ValueError as err:
        print(f'Dimensionality reduction failed: {err}.')

    if classes is not None:
        cf.set_dgroups(classes)
        embeddf['class'] = cf.get_dgroup_names(feature_matrix.index)

    if not plot_nas:
        embeddf = embeddf[embeddf['class'] != 'N/A']

    fig, _ = plt.subplots(1,1)
    try:
        if include_size:
            sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,hue='class',size='n_procs',alpha=.8,sizes=(50,200),**scatterplot_kwargs)
        else:
            sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,hue='class',alpha=.8,**scatterplot_kwargs)
    except ValueError as err:
        if include_size:
            sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,size='n_procs',alpha=.8,sizes=(50,200),**scatterplot_kwargs)
        else:
            sns.scatterplot(data=embeddf,x=xlabel,y=ylabel,alpha=.8,**scatterplot_kwargs)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.legend(bbox_to_anchor=(0.5, 1.05), loc='lower center', borderaxespad=0., ncol=3)
    plt.legend()

    return fig, embeddf

def relate(cf, stat0, stat1, absolute=False, classes=None, include_total=True, method='spearman'):

    if classes is not None:
        cf.set_dgroups(classes)

    s0mdf = cf.mdf[stat0].reset_index()
    smdf = s0mdf[s0mdf['Source statistic'] == stat1]

    name = f'r({stat0},{stat1})'
    smdf = smdf.rename(columns={stat0: name})

    if absolute:
        smdf[name] = smdf[name].abs()

    _, ax = plt.subplots()
    if classes is not None:
        smdf['class'] = cf.get_dgroup_names(smdf['Dataset'])
        smdf_l = smdf[smdf['class'] != 'N/A']
        sns.histplot(smdf_l,x=name,hue='class',stat='probability',common_norm=True, multiple='stack')
        if include_total:
            sns.histplot(smdf,x=name,stat='probability',element="step",alpha=.2)
            ax.axvline(smdf[name].mean(),c='k',ls='--')
        ax.axvline(smdf_l[name].mean(),c='k',ls='-')
    else:
        sns.histplot(smdf,x=name,stat='probability',element="step",common_norm=True, multiple='stack')
        ax.axvline(smdf[name].mean(),c='k',ls='--')
    return smdf

# For now just take in the dataframe computed by cf.correlation_matrix()
def concensusmap(df,n_clusters=8):
    cdf = pd.DataFrame(columns=df.columns)
    for _, new_df in df.groupby(level=0):
        kmeans = cluster.KMeans(n_clusters).fit(new_df.values)
        ndf = pd.DataFrame(data=kmeans.labels_,index=new_df.name,columns=cdf.columns)
        cdf = cdf.append(ndf)