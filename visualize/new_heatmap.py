import torch
import numpy as np
import pickle
import pdb 
from tqdm import tqdm
from torchvision.ops.boxes import batched_nms, box_iou
import sys
import matplotlib.pyplot as plt
import seaborn as sns

pkl_file = '../new_embedding.p'
with open(pkl_file, 'rb') as f:
    new_embeddings = pickle.load(f)

title = 'direct airplane'
heat_mapp = new_embedding @ new_embedding.t()

sns.heatmap(heat_mapp[:, :].cpu().numpy(), annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title(title, fontsize=16)
plt.savefig(f'heatmap_{title}.png')