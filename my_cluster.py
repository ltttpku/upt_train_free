from genericpath import exists
import sklearn
import pickle
import pdb
import torch
from torchvision.ops.boxes import batched_nms, box_iou
from sklearn.cluster import KMeans, MiniBatchKMeans
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from tqdm import tqdm
import io
from hico_list import hico_verb_object_list
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import hdbscan

# file1 = 'union_embeddings_cachemodel_clipcrops.p'

file1 = 'union_embeddings_cachemodel_crop_padding_zeros.p'
## CODE START
rgb_base_dir = 'hicodet/hico_20160224_det/images/train2015'
rgbs_file = 'all_rgbs_in_trainingset.p'
output_base_dir = 'cluster_output'
black_and_white_cnt = 0
# annotation = pickle.load(open(file11,'rb'))
if not exists(output_base_dir):
    os.makedirs(output_base_dir)
font = ImageFont.truetype('/scratch/shared/beegfs/yangl/leiting/fonts/arial.ttf', size=40)
print('font:', font)

def save_tgt_rgbs(tgt_rgbs_lst_lst, prefix='object'):
    for i, cluster_rgb_lst in enumerate(tgt_rgbs_lst_lst):
        if i >= 15:
            break
        ## shuffle cluster_rgb_lst
        np.random.shuffle(cluster_rgb_lst)
        cluster_rgb_lst = [np.array(Image.fromarray(rgb).resize((224,224))) for rgb in cluster_rgb_lst[:10]]
        matrix = np.concatenate(cluster_rgb_lst, axis=1)
        img = Image.fromarray(matrix.astype('uint8'))
        img.save(os.path.join(output_base_dir, prefix + '_cluster_'+str(i)+'.png'))
## code end

def get_instance_features(file1,class_nums=600):
    global black_and_white_cnt, font
    annotation = pickle.load(open(file1,'rb'))
    rgbs = pickle.load(open(rgbs_file,'rb'))
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
        rgb = rgbs[file_n]
        if len(rgb.shape) == 2:
            black_and_white_cnt += 1
            rgb = np.stack([rgb,rgb,rgb], axis=2) ## convert black,white to rgb
        image = Image.fromarray(rgb).resize((224,224))

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
            ## CODE START
            # pdb.set_trace()
            image2 = image.copy()
            draw = ImageDraw.Draw(image2)
            draw.rectangle(anno['boxes_o'][i].tolist(), outline='red', width=3)
            draw.text((anno['boxes_o'][i][0], anno['boxes_o'][i][1]), hico_verb_object_list[anno['hois'][i]][1], fill='red', font = font)
            obj_images[v].append(np.array(image2))
            
            image3 = image.copy()
            draw = ImageDraw.Draw(image3)
            draw.rectangle(anno['boxes_h'][i].tolist(), outline='red', width=3)
            draw.text((anno['boxes_h'][i][0], anno['boxes_h'][i][1]), hico_verb_object_list[anno['hois'][i]][0], fill='red', font = font)
            hum_images[v].append(np.array(image3))
            ## CODE END

            # union_embeddings[v].append(union_features[i])
            obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i])) ## normalize every vectors
            hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))
            each_filenames[v].append(file_n)
            sample_indexes[v].append(i)
            # add iou
            verbs_iou[v].append(ious[i])
    return obj_embeddings, hum_embeddings, obj_images, hum_images


def naive_cluster_centers(embeddings, center_num):
    centers = []
    for i in range(center_num):
        centers.append(torch.mean(torch.as_tensor(embeddings[i]), dim=0, keepdim=False))
    centers = torch.stack(centers, dim=0)
    return centers

_C_obj, _C_hum = 80, 117
object_embeddings117, human_embeddings117, obj_images, hum_images = get_instance_features(file1, class_nums=117)
object_embeddings80, human_embeddings80, obj_images, hum_images = get_instance_features(file1, class_nums=80)

cluster_centers_hum = naive_cluster_centers(human_embeddings117, center_num=117)
cluster_centers_obj = naive_cluster_centers(object_embeddings80, center_num=80)

object_embeddings, human_embeddings, obj_images, hum_images = get_instance_features(file1, class_nums=600)

## CODE START    
    # print('black_and_white_cnt:', black_and_white_cnt)
    # pdb.set_trace()
    # all_obj_rgbs = []
    # all_hum_rgbs = []
    # for objrgb in obj_images:
    #     all_obj_rgbs.extend(objrgb)
    # for humrgb in hum_images:
    #     all_hum_rgbs.extend(humrgb)
    # ## CODE END

    # all_object_embs = []
    # for obj in object_embeddings:
    #     all_object_embs.extend(obj)

    # all_human_embs = []
    # for hum in human_embeddings:
    #     all_human_embs.extend(hum)
    # all_object_embs = torch.as_tensor(all_object_embs)
    # all_object_embs /= all_object_embs.norm(dim=-1, keepdim=True)
    # all_human_embs = torch.as_tensor(all_human_embs)
    # all_human_embs /= all_human_embs.norm(dim=-1, keepdim=True)

    # ## ----------------------------------------------
    # print("kmeans for objects")

    # pdb.set_trace()
    # # model = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
    # # model.fit(all_object_embs)
    # # obj_labels_ = model.labels_
    # # cluster_centers_obj = np.array([model.weighted_cluster_centroid(label) for label in range(max(obj_labels_) + 1)])

    # _C_obj = 80
    # # kmeans = MiniBatchKMeans(n_clusters=_C_obj, random_state=1234)
    # kmeans = KMeans(n_clusters=_C_obj, random_state=1234) 
    # kmeans.fit(all_object_embs)
    # cluster_centers_obj = torch.from_numpy(kmeans.cluster_centers_)
    # # silhouette_coefficient = silhouette_score(all_object_embs, kmeans.labels_, metric='euclidean')
    # # print(silhouette_coefficient)
    # # if silhouette_coefficient > 0.5:
    # #     break

    # # silhouette_coefficients = []
    # # sse = []
    # # _start = 50
    # # _end = 250
    # # _step = 10
    # # for _C_obj in tqdm(range(_start, _end, _step)):
    # #     # kmeans = KMeans(n_clusters=_C_obj, random_state=1234)
    # #     kmeans = MiniBatchKMeans(n_clusters=_C_obj, random_state=1234)
    # #     kmeans.fit(all_object_embs)
    # #     silhouette_coefficients.append(silhouette_score(all_object_embs, kmeans.labels_, metric='euclidean'))
    # #     sse.append(kmeans.inertia_)

    # # plt.plot(range(_start, _end, _step), silhouette_coefficients)
    # # plt.xticks(range(_start, _end, _step))
    # # plt.xlabel('number of clusters')
    # # plt.ylabel('silhouette coefficient')
    # # plt.savefig('silhouette_obj.png')
    # # ## clear plt
    # # plt.clf()

    # # plt.plot(range(_start, _end, _step), sse)
    # # plt.xticks(range(_start, _end, _step))
    # # plt.xlabel('number of clusters')
    # # plt.ylabel('sse')
    # # plt.savefig('sse_obj.png')
    # # plt.clf()

    # # save #num rgbs for each cluster
    # tgt_obj_rgbs = [[] for i in range(_C_obj)]
    # for i, label in enumerate(kmeans.labels_):
    #     tgt_obj_rgbs[label].append(all_obj_rgbs[i])
    # save_tgt_rgbs(tgt_obj_rgbs, prefix='object')

    # ## -------------------------------------------------------
    # print('kneams for humans')
    # _C_hum = 30
    # kmeans = KMeans(n_clusters=_C_hum, random_state=1234)
    # kmeans.fit(all_human_embs)

    # # silhouette_coefficients = []
    # # sse = []
    # # _start = 10
    # # _end = 150
    # # _step = 10
    # # for _C_hum in tqdm(range(_start, _end, _step)):
    # #     kmeans = KMeans(n_clusters=_C_hum, random_state=1234)
    # #     kmeans.fit(all_human_embs)
    # #     silhouette_coefficients.append(silhouette_score(all_human_embs, kmeans.labels_, metric='euclidean'))
    # #     sse.append(kmeans.inertia_)

    # # plt.plot(range(_start, _end, _step), silhouette_coefficients)
    # # plt.xticks(range(_start, _end, _step))
    # # plt.xlabel('number of clusters')
    # # plt.ylabel('silhouette coefficient')
    # # plt.savefig('silhouette_hum.png')
    # # ## clear plt
    # # plt.clf()

    # # plt.plot(range(_start, _end, _step), sse)
    # # plt.xticks(range(_start, _end, _step))
    # # plt.xlabel('number of clusters')
    # # plt.ylabel('sse')
    # # plt.savefig('sse_hum.png')
    # # plt.clf()

    # cluster_centers_hum = torch.from_numpy(kmeans.cluster_centers_)
    # tgt_hum_rgbs = [[] for i in range(_C_hum)]
    # for i, label in enumerate(kmeans.labels_):
    #     tgt_hum_rgbs[label].append(all_hum_rgbs[i])
    # save_tgt_rgbs(tgt_hum_rgbs, prefix='human')
    ## CODE END

pdb.set_trace()
cluster_centers_hum = cluster_centers_hum.float()
cluster_centers_obj = cluster_centers_obj.float()
cluster_centers_hum = cluster_centers_hum/cluster_centers_hum.norm(dim=-1,keepdim=True)
cluster_centers_obj = cluster_centers_obj/cluster_centers_obj.norm(dim=-1,keepdim=True)

hoi_scores = []
indexes = torch.range(0, 600)
for idx, h_emb, o_emb in zip(indexes, human_embeddings, object_embeddings):
    h_emb = torch.as_tensor(h_emb).float()
    o_emb = torch.as_tensor(o_emb).float()
    h_emb = h_emb/h_emb.norm(dim=-1,keepdim=True)
    o_emb = o_emb/o_emb.norm(dim=-1,keepdim=True)
    print(idx, h_emb.shape, o_emb.shape)

    score_o = o_emb @ cluster_centers_obj.t()
    score_o /= score_o.norm(dim=-1,keepdim=True)
    score_h = h_emb @ cluster_centers_hum.t()
    score_h /= score_h.norm(dim=-1,keepdim=True)
    score = torch.cat((score_o, score_h), dim=1).mean(dim=0) ## dim: C1 + C2
    # score_mean = ((score_h + score_o)/2).mean(dim=0)
    hoi_scores.append(score)


hoi_scores = torch.stack(hoi_scores)

dicts = {}
dicts['cluster_centers_hum'] = cluster_centers_hum
dicts['cluster_centers_obj'] = cluster_centers_obj
dicts['hoi_scores'] = hoi_scores

with open(f'clusters_{_C_obj}_{_C_hum}_padding_zeros.p','wb') as f:
    pickle.dump(dicts, f)