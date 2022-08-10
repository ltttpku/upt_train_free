import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn
import pickle
import random
import pdb
import matplotlib.cm as cm 

pkl_file = '../embs_ye.p'
with open(pkl_file, 'rb') as f:
    new_embeddings = pickle.load(f)
new_embeddings = new_embeddings/new_embeddings.norm(dim=-1,keepdim=True)
# title = 'load airplane'
title = 'lick bottle'

# origin_idx = torch.arange(0, new_embeddings.shape[0], 2)

origin_idx = torch.arange(0,7)

tsne = TSNE(n_components=2, perplexity=30, n_iter=3000)
tsne.fit_transform(new_embeddings)
embeddings_tsne = tsne.embedding_
pdb.set_trace()

colors = cm.rainbow(np.linspace(0, 1, 2))
plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1, marker='o', label='augmented', color=colors[0])
plt.scatter(embeddings_tsne[origin_idx, 0], embeddings_tsne[origin_idx, 1], s=1, marker='o', label='real', color=colors[1])

plt.title(title)
plt.legend(loc='best')
plt.savefig(f'origin_vs_augmented_{title}_ye.png')
plt.clf()
plt.close()

# chosen_idx = torch.tensor([1871, 1687, 1679,  203,  459,  453,  845,  921, 1248, 1106, 1084, 1946,
#         1894, 1536, 1932, 1306,  108,  130,  736,  506,  412,  274,  190,  456,
#          776,  828,  678, 1202, 1698, 1834, 1594, 1952])
# plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1, marker='o', label='augmented', color=colors[0])
# plt.scatter(embeddings_tsne[chosen_idx, 0], embeddings_tsne[chosen_idx, 1], s=1, marker='o', label='chosen', color=colors[1])
# plt.legend()
# plt.savefig('chosen_vs_augmented.png')