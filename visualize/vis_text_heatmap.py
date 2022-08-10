import torch
import numpy as np
import pickle
import pdb 
from tqdm import tqdm
from torchvision.ops.boxes import batched_nms, box_iou
import sys
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('..')
from hico_list import *
from hico_text_label import *

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

def get_text_embedding(sentence_lst):
    pdb.set_trace()
    import clip
    
    clip_model, _ = clip.load("ViT-B/32", device=device)
    text_inputs = torch.cat([clip.tokenize(v) for v in sentence_lst])
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text_inputs.to(device))
    text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
    print('text embedding shape:', text_embedding.shape)
    return text_embedding


file1 = '../union_embeddings_cachemodel_crop_padding_zeros.p'
device = "cuda" if torch.cuda.is_available() else "cpu"
_C_obj, _C_hum = 80, 117
branch = 'human' ## human, object

if branch == 'human':
    object_embeddings117, human_embeddings117, obj_images, hum_images = get_instance_features(file1, class_nums=117)

    text_embedding = get_text_embedding([sentence.replace('the', 'an') for sentence in hico_verbs_sentence])
    score_lst = []
    for i, hum_emb in enumerate(human_embeddings117):
        hum_emb = torch.as_tensor(hum_emb).to(device)
        hum_emb = hum_emb / hum_emb.norm(dim=-1, keepdim=True) ## e.g., 32 x 512
        
        text_emb = text_embedding.to(device) ## 117 x 512
        scores = hum_emb.mm(text_emb.t())
        scores = scores.mean(dim=0) ## 117
        score_lst.append(scores)

    # pdb.set_trace()
    heat_mapp = torch.stack(score_lst, dim=0) ## 117 x 117
    ## visualize confusion matrix
    sns.heatmap(heat_mapp[:, :].cpu().numpy(), annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.ylabel('visual feature')
    plt.xlabel('text embddding')
    plt.title('human-heatmap', fontsize=20)
    ## save heatmap
    plt.savefig('human_heatmap.png')

elif branch == 'object':
    object_embeddings80, human_embeddings80, obj_images, hum_images = get_instance_features(file1, class_nums=80)

    text_embedding = get_text_embedding([pair[1] for pair in hico_obj_text_label])
    score_lst = []
    for i, hum_emb in enumerate(object_embeddings80):
        hum_emb = torch.as_tensor(hum_emb).to(device)
        hum_emb = hum_emb / hum_emb.norm(dim=-1, keepdim=True) ## e.g., 32 x 512
        
        text_emb = text_embedding.to(device) ## 80 x 512
        scores = hum_emb.mm(text_emb.t())
        scores = scores.mean(dim=0) ## 80
        score_lst.append(scores)

    # pdb.set_trace()
    heat_mapp = torch.stack(score_lst, dim=0) ## 80(visual) x 81(Text)
    print('object heatmap shape:', heat_mapp.shape)
    ## visualize confusion matrix
    sns.heatmap(heat_mapp[:, :].cpu().numpy(), annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.ylabel('visual feature')
    plt.xlabel('text embddding')
    plt.title('object-heatmap', fontsize=20)
    ## save heatmap
    plt.savefig('object_heatmap.png')
