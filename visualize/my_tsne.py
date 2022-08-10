import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn
import pickle
import random
import pdb
import matplotlib.cm as cm 

hico_verbs = ['adjust', 'assemble', 'block', 'blow', 'board', 'break', 'brush with', 'buy', 
 'carry', 'catch', 'chase', 'check', 'clean', 'control', 'cook', 'cut', 'cut with', 'direct', 
 'drag', 'dribble', 'drink with', 'drive', 'dry', 'eat', 'eat at', 'exit', 'feed', 'fill', 
 'flip', 'flush', 'fly', 'greet', 'grind', 'groom', 'herd', 'hit', 'hold', 'hop on', 'hose', 
 'hug', 'hunt', 'inspect', 'install', 'jump', 'kick', 'kiss', 'lasso', 'launch', 'lick', 'lie on', 
 'lift', 'light', 'load', 'lose', 'make', 'milk', 'move', 'no interaction', 'open', 'operate', 'pack', 
 'paint', 'park', 'pay', 'peel', 'pet', 'pick', 'pick up', 'point', 'pour', 'pull', 'push', 'race', 'read', 
 'release', 'repair', 'ride', 'row', 'run', 'sail', 'scratch', 'serve', 'set', 'shear', 'sign', 'sip', 'sit at', 
 'sit on', 'slide', 'smell', 'spin', 'squeeze', 'stab', 'stand on', 'stand under', 'stick', 'stir', 'stop at', 
 'straddle', 'swing', 'tag', 'talk on', 'teach', 'text on', 'throw', 'tie', 'toast', 'train', 'turn', 'type on', 
 'walk', 'wash', 'watch', 'wave', 'wear', 'wield', 'zip']

pkl_file = 'human_embeddings_117_random_20.p'
# pkl_file = 'object_embeddings_80_random_20.p'
with open(pkl_file, 'rb') as f:
    dct = pickle.load(f)
embeddings = dct['embeddings']
sample_idx = dct['sample_idx']
pdb.set_trace()

embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000)
tsne.fit_transform(embeddings)
embeddings_tsne = tsne.embedding_

start_idx = 5
category_num = 5
assert category_num + start_idx < sample_idx.shape[0]
colors = cm.rainbow(np.linspace(0, 1, category_num))
for start_idx in range(0, 25, 5):
    for ii, end_idx in enumerate(sample_idx[start_idx:start_idx+category_num]):
        print(end_idx)
        i = ii + start_idx
        if i == 0:
            plt.scatter(embeddings_tsne[:end_idx, 0], embeddings_tsne[:end_idx, 1], s=5, marker='o', label=str(hico_verbs[i]), color=colors[ii])
        else:
            plt.scatter(embeddings_tsne[sample_idx[i-1]:sample_idx[i], 0], embeddings_tsne[sample_idx[i-1]:sample_idx[i], 1], s=5, marker='o', label=str(hico_verbs[i]), color=colors[ii])
    # plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)
    plt.legend(loc='best')
    plt.title(pkl_file.split('.')[0] + '_start_idx=' + str(start_idx))
    ## save figure
    plt.savefig( pkl_file.split('.')[0] + '_start_idx=' + str(start_idx) + '.png')
    ## clear plt figure
    plt.clf()
    plt.close()