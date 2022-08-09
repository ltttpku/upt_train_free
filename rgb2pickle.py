import sklearn
import pickle
import pdb
import torch
from torchvision.ops.boxes import batched_nms, box_iou
from sklearn.cluster import KMeans
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import io

file1 = 'union_embeddings_cachemodel_crop_padding_zeros.p'
outputfile = 'all_rgbs_in_trainingset.p'
rgb_base_dir = 'hicodet/hico_20160224_det/images/train2015'
class_nums=600

annotation = pickle.load(open(file1,'rb'))
filenames = list(annotation.keys())
tgtdct = {}

for file_n in tqdm(filenames):
    ## CODE START
    rgb_path = os.path.join(rgb_base_dir,file_n)
    image = Image.open(rgb_path).resize((224,224))
    tgtdct[file_n] = np.array(image)
    ## CODE END

# pdb.set_trace()
with open(outputfile,'wb') as f:
    pickle.dump(tgtdct,f)