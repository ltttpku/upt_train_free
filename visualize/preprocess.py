import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sklearn
import pickle
import pdb 
from tqdm import tqdm
from torchvision.ops.boxes import batched_nms, box_iou



def get_instance_features(file1,class_nums=600):
    global black_and_white_cnt, font
    annotation = pickle.load(open(file1,'rb'))
    # rgbs = pickle.load(open(rgbs_file,'rb'))
    # if category == 'verb':
    categories = class_nums
    union_embeddings = [[] for i in range(categories)]
    obj_embeddings = [[] for i in range(categories)]
    hum_embeddings = [[] for i in range(categories)]
    ## CODE START
    obj_images = [[] for i in range(categories)]
    hum_images = [[] for i in range(categories)]
    ## CODE END
    filenames = list(annotation.keys())
    verbs_iou = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
    # hois_iou = [[] for i in range(len(hois))]
    # filenames = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
    each_filenames = [[] for i in range(categories)]
    sample_indexes = [[] for i in range(categories)]

    print('total num of imgs:', len(filenames))
    for file_n in tqdm(filenames):
        anno = annotation[file_n]
        if categories == 117: verbs = anno['verbs']
        elif categories == 80: verbs = anno['objects']
        else: verbs = anno['hois']

        union_features = anno['union_features']
        object_features = anno['object_features']
        huamn_features = anno['huamn_features']
        ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
        # pdb.set_trace()
        if len(verbs) == 0:
            print(file_n)

        for i, v in enumerate(verbs):
            # union_embeddings[v].append(union_features[i])
            obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i])) ## normalize every vectors
            hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))
            each_filenames[v].append(file_n)
            sample_indexes[v].append(i)
            # add iou
            verbs_iou[v].append(ious[i])
    return obj_embeddings, hum_embeddings, obj_images, hum_images


def random_choose_embeddings(embeddings, sample_num=3):
    res = []
    sample_idx = []
    for i in range(len(embeddings)):
        embedding = torch.as_tensor(embeddings[i])
        if embedding.shape[0] >= sample_num:
            res.append(embedding[torch.randint(0, embedding.shape[0], (sample_num,))])
            sample_idx.append(sample_num)
        else:
            res.append(embedding)
            sample_idx.append(embedding.shape[0])

    sample_idx = torch.as_tensor(sample_idx)
    sample_idx = torch.cumsum(sample_idx, dim=-1)
    res = torch.cat(res, dim=0)
    return res, sample_idx

file1 = '../union_embeddings_cachemodel_crop_padding_zeros.p'
_C_obj, _C_hum = 80, 117
sample_num_per_class = 20
branch = 'object' ## human object

if branch == 'human':
    object_embeddings117, human_embeddings117, obj_images, hum_images = get_instance_features(file1, class_nums=117)

    embeddings, sample_idx = random_choose_embeddings(human_embeddings117, sample_num=sample_num_per_class)

    ## save to pickle
    dct = {'embeddings': embeddings, 'sample_idx': sample_idx}
    pickle.dump(dct, open(f'human_embeddings_117_random_{sample_num_per_class}.p', 'wb'))

elif branch == 'object':
    object_embeddings80, human_embeddings80, obj_images, hum_images = get_instance_features(file1, class_nums=80)

    embeddings, sample_idx = random_choose_embeddings(object_embeddings80, sample_num=sample_num_per_class)
    pdb.set_trace()
    ## save to pickle
    dct = {'embeddings': embeddings, 'sample_idx': sample_idx}
    pickle.dump(dct, open(f'object_embeddings_80_random_{sample_num_per_class}.p', 'wb'))