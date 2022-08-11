import pickle 
import torch
from torchvision.ops.boxes import batched_nms, box_iou
import numpy as np
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
import pdb
import  hico_dataset_list  as hico_list
unseen_rare_first = hico_list.hico_unseen_index['rare_first']
from sklearn.manifold import TSNE
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import sklearn
visualization = False
use_inter_swap = True
if not visualization:
    HOI_IDX_TO_OBJ_IDX = [
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                    14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                    39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                    56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                    60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                    16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                    58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                    6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                    24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                    34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                    73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                    55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                    67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                    54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                    20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                    23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                    52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                    43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                    49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                    53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                    48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                    36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                    31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                    32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                    28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                    38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                    61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                    7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                    40, 40, 22, 22, 22, 22, 22
                ]
    HOI_IDX_TO_ACT_IDX = [
            4, 17, 25, 30, 41, 52, 76, 87, 111, 57, 8, 36, 41, 43, 37, 62, 71, 75, 76,
            87, 98, 110, 111, 57, 10, 26, 36, 65, 74, 112, 57, 4, 21, 25, 41, 43, 47,
            75, 76, 77, 79, 87, 93, 105, 111, 57, 8, 20, 36, 41, 48, 58, 69, 57, 4, 17,
            21, 25, 41, 52, 76, 87, 111, 113, 57, 4, 17, 21, 38, 41, 43, 52, 62, 76,
            111, 57, 22, 26, 36, 39, 45, 65, 80, 111, 10, 57, 8, 36, 49, 87, 93, 57, 8,
            49, 87, 57, 26, 34, 36, 39, 45, 46, 55, 65, 76, 110, 57, 12, 24, 86, 57, 8,
            22, 26, 33, 36, 38, 39, 41, 45, 65, 78, 80, 98, 107, 110, 111, 10, 57, 26,
            33, 36, 39, 43, 45, 52, 37, 65, 72, 76, 78, 98, 107, 110, 111, 57, 36, 41,
            43, 37, 62, 71, 72, 76, 87, 98, 108, 110, 111, 57, 8, 31, 36, 39, 45, 92,
            100, 102, 48, 57, 8, 36, 38, 57, 8, 26, 34, 36, 39, 45, 65, 76, 83, 110,
            111, 57, 4, 21, 25, 52, 76, 87, 111, 57, 13, 75, 112, 57, 7, 15, 23, 36,
            41, 64, 66, 89, 111, 57, 8, 36, 41, 58, 114, 57, 7, 8, 15, 23, 36, 41, 64,
            66, 89, 57, 5, 8, 36, 84, 99, 104, 115, 57, 36, 114, 57, 26, 40, 112, 57,
            12, 49, 87, 57, 41, 49, 87, 57, 8, 36, 58, 73, 57, 36, 96, 111, 48, 57, 15,
            23, 36, 89, 96, 111, 57, 3, 8, 15, 23, 36, 51, 54, 67, 57, 8, 14, 15, 23,
            36, 64, 89, 96, 111, 57, 8, 36, 73, 75, 101, 103, 57, 11, 36, 75, 82, 57,
            8, 20, 36, 41, 69, 85, 89, 27, 111, 57, 7, 8, 23, 36, 54, 67, 89, 57, 26,
            36, 38, 39, 45, 37, 65, 76, 110, 111, 112, 57, 39, 41, 58, 61, 57, 36, 50,
            95, 48, 111, 57, 2, 9, 36, 90, 104, 57, 26, 45, 65, 76, 112, 57, 36, 59,
            75, 57, 8, 36, 41, 57, 8, 14, 15, 23, 36, 54, 57, 8, 12, 36, 109, 57, 1, 8,
            30, 36, 41, 47, 70, 57, 16, 36, 95, 111, 115, 48, 57, 36, 58, 73, 75, 109,
            57, 12, 58, 59, 57, 13, 36, 75, 57, 7, 15, 23, 36, 41, 64, 66, 91, 111, 57,
            12, 36, 41, 58, 75, 59, 57, 11, 63, 75, 57, 7, 8, 14, 15, 23, 36, 54, 67,
            88, 89, 57, 12, 36, 56, 58, 57, 36, 68, 99, 57, 8, 14, 15, 23, 36, 54, 57,
            16, 36, 58, 57, 12, 75, 111, 57, 8, 28, 32, 36, 43, 67, 76, 87, 93, 57, 0,
            8, 36, 41, 43, 67, 75, 76, 93, 114, 57, 0, 8, 32, 36, 43, 76, 93, 114, 57,
            36, 48, 111, 85, 57, 2, 8, 9, 19, 35, 36, 41, 44, 67, 81, 84, 90, 104, 57,
            36, 94, 97, 57, 8, 18, 36, 39, 52, 58, 60, 67, 116, 57, 8, 18, 36, 41, 43,
            49, 52, 76, 93, 87, 111, 57, 8, 36, 39, 45, 57, 8, 36, 41, 99, 57, 0, 15,
            36, 41, 70, 105, 114, 57, 36, 59, 75, 57, 12, 29, 58, 75, 87, 93, 111, 57,
            6, 36, 111, 57, 42, 75, 94, 97, 57, 17, 21, 41, 52, 75, 76, 87, 111, 57, 8,
            36, 53, 58, 75, 82, 94, 57, 36, 54, 61, 57, 27, 36, 85, 106, 48, 111, 57,
            26, 36, 65, 112, 57
        ]
    file1 = 'union_embeddings_cachemodel_crop_padding_zeros_vitb16.p'
    file_bbox = 'save_bboxes_ye.p'
    bboxes_gt = pickle.load(open(file_bbox,'rb'))

    class_nums = 600
    annotation = pickle.load(open(file1,'rb'))
    # if category == 'verb':
    categories = class_nums
    union_embeddings = [[] for i in range(categories)]
    obj_embeddings = [[] for i in range(categories)]
    hum_embeddings = [[] for i in range(categories)]
    filenames = list(annotation.keys())
    verbs_iou = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
    # hois_iou = [[] for i in range(len(hois))]
    # filenames = [[] for i in range(class_nums)] # contain 600hois or 117 verbs
    each_filenames = [[] for i in range(categories)]
    sample_indexes = [[] for i in range(categories)]

    # play for unseen or inter swap
    verbs_human_feat = [[] for i in range(117)]
    verbs_human_area = [[] for i in range(117)]

    gt_pair_hum_obj_area = [[] for i in range(categories)]
    gt_pair_hum_obj_area_ratio = [[] for i in range(categories)]

    obj_belong_to_object_pool = [[] for i in range(categories)]
    object_feat = [[] for i in range(80)]
    object_ref_area = [[] for i in range(80)]
    human_ref_area =  [[] for i in range(categories)]


    object_ref_area_ratio = [[] for i in range(80)]



    others_hum = [[] for i in range(categories)]

   
    # others_hum_area = [[] for i in range(categories)]
    count = 0

    for file_n in filenames:
        anno = annotation[file_n]
        if categories == 117: verbs = anno['verbs']
        else: verbs = anno['hois']

        union_features = anno['union_features']
        object_features = anno['object_features']
        # pdb.set_trace()
        huamn_features = anno['huamn_features']
        # gt_hum_boxes = torch.as_tensor(anno['boxes_h'])
        # gt_obj_boxes = torch.as_tensor(anno['boxes_o'])    
        # pdb.set_trace()
        gt_hum_boxes = torch.as_tensor(bboxes_gt[file_n]['boxes_h'])
        gt_obj_boxes = torch.as_tensor(bboxes_gt[file_n]['boxes_o']) 


        ious = torch.diag(box_iou(gt_hum_boxes, gt_obj_boxes))
        # pdb.set_trace()
        x, y = torch.nonzero(torch.min(
            box_iou(gt_hum_boxes, gt_hum_boxes),
            box_iou(gt_obj_boxes, gt_obj_boxes),
            ) >= 0.5).unbind(1)
        # 

    #     pdb.set_trace()
        orig_verbs = anno['verbs']
        objects_label = torch.as_tensor(anno['objects'])

        
        area1 = (gt_hum_boxes[:,2]-gt_hum_boxes[:,0]) * (gt_hum_boxes[:,3]-gt_hum_boxes[:,1])
        area2 = (gt_obj_boxes[:,2]-gt_obj_boxes[:,0]) * (gt_obj_boxes[:,3]-gt_obj_boxes[:,1])
        area_ratio =   area2/ area1                  
        if len(verbs) == 0:
            print(file_n)
        # if torch.sum(objects_label[0]==objects_label) != objects_label.shape[0]:
        #     print(objects_label)
        #     pdb.set_trace()
        count+=1
        for i, v in enumerate(verbs):

            union_embeddings[v].append(union_features[i] / np.linalg.norm(union_features[i]))
            obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i]))
            hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))
            each_filenames[v].append(file_n)
            sample_indexes[v].append(i)
            gt_pair_hum_obj_area[v].append((area1[i],area2[i],area1[i]/area2[i]))
    #         object_ref_area_ratio[v].append(area_ratio[i])
            
            
            
            
            
            ind_x = torch.where(x==i)[0]
            ind_y = torch.where(y[ind_x]!=i)[0]
            object_pair = torch.where(objects_label[i]==objects_label[ind_y])[0]
            ind_y_new = ind_y[object_pair]
            if len(ind_y_new) == 1:
                # ind_y_new = ind_y_new[None,]
                others_hum[v].append(torch.as_tensor(huamn_features[ind_y_new][None,]))
                # pdb.set_trace()
            else:
                others_hum[v].append(torch.as_tensor(huamn_features[ind_y_new]))
                
            object_ref_area[objects_label[i]].append(area2[i])
            human_ref_area[v].append(area1[ind_y_new])
            
            verbs_iou[v].append(ious[i])
    #         if v in self.unseen_nonrare_first and self.unseen_setting:
    #             # pdb.set_trace()
    #             continue
            verbs_human_feat[orig_verbs[i]].append(torch.as_tensor(huamn_features[i] / np.linalg.norm(huamn_features[i])))
            verbs_human_area[orig_verbs[i]].append(area1[i])
            obj_belong_to_object_pool[v].append(len(object_feat[objects_label[i]]))
            object_feat[objects_label[i]].append(object_features[i] / np.linalg.norm(object_features[i]))
            
            # add iou
    cache_models = []
    one_hots = []
    each_lens = []
    indexes = np.arange(len(union_embeddings))
    save_all_list = []
    topk = 5
    all_count = []
    
    for i, hum_emb, obj_emb, embeddings in zip(indexes, hum_embeddings, obj_embeddings, union_embeddings):
        
        save_one_hum = []
        save_one_obj = []
        new_count = 0
        range_lens = np.arange(len(embeddings))
        hum_emb = torch.as_tensor(hum_emb)
        obj_emb = torch.as_tensor(obj_emb)
        hum_area = torch.stack([gt_pair[0] for gt_pair in gt_pair_hum_obj_area[i]])
        obj_area = torch.stack([gt_pair[1] for gt_pair in gt_pair_hum_obj_area[i]])
        hum_obj_ratio = torch.stack([gt_pair[2] for gt_pair in gt_pair_hum_obj_area[i]])

        lens = len(hum_emb)

        indexes = torch.arange(0,lens)[:,None]

        if len(hum_emb) < 100:
            if use_inter_swap:
                # pdb.set_trace()
                ref_area = torch.stack(verbs_human_area[HOI_IDX_TO_ACT_IDX[i]])
                ref_hum = torch.stack(verbs_human_feat[HOI_IDX_TO_ACT_IDX[i]])
                ref_hum = ref_hum/ ref_hum.norm(dim=-1, keepdim=True)
            else:
                ref_area = torch.cat(human_ref_area[i])
                ref_hum = torch.cat(others_hum[i])
                ref_hum = ref_hum/ ref_hum.norm(dim=-1, keepdim=True)
            new_hum = torch.cat([hum_emb,ref_hum])
            sim_ = new_hum @ new_hum.t()
        #     pdb.set_trace()
            new_hum_area = torch.cat([hum_area, ref_area],dim=0)
            
            # object 
            
            
            object_cls = HOI_IDX_TO_OBJ_IDX[i]
            object_ref_embs = torch.stack([torch.as_tensor(obj) for obj in object_feat[object_cls]])
            sim_obj = obj_emb @ object_ref_embs.t()
            
            
            
            ref_obj_area = torch.stack(object_ref_area[object_cls])
            object_indexes_pool = obj_belong_to_object_pool[i]
            
            
            
            for k, sim in enumerate(sim_[:lens]):
                
                # rule1: human simialrity
                # sort_indexes = sim.sort(descending=True)[1]
                # valid_indexes = sort_indexes[torch.where(sort_indexes != k)[0]]
                # sample_lens = min(len(valid_indexes),5)
                # sample_hum_feats = new_hum[valid_indexes][:sample_lens]
                
                # object baneen 1
                # obj_area_ = obj_area[k]
                # obj_ratio_ = torch.abs(ref_obj_area/obj_area_  -1) 

                # sort_obj_indexes = obj_ratio_.sort()[1]
                # current_obj = object_indexes_pool[k]

                # valid_obj_indexes = sort_obj_indexes[torch.where(sort_obj_indexes != current_obj)[0]]
                # sample_obj_lens = min(len(valid_obj_indexes),5)
                # sample_obj_feats = object_ref_embs[valid_obj_indexes][:sample_obj_lens]
                # pdb.set_trace()
                # # banben 2
                # obj_area_ = obj_area[k]
                # obj_ratio_ = torch.abs(ref_obj_area/obj_area_  -1) <0.1
                # current_obj = object_indexes_pool[k]
                # obj_ratio_[current_obj] = False
                # sim_obj_embs = sim_obj[k]
                # sim_obj_embs = sim_obj_embs * obj_ratio_
                # valid_value, valid_obj_indexes = sim_obj_embs.sort(descending=True)
                # valid_indexes_for_obj = torch.where(valid_value>0.5)[0]
                # valid_obj_indexes = valid_obj_indexes[valid_indexes_for_obj]
                # sample_obj_lens = min(len(valid_obj_indexes),5)
                # sample_obj_feats = object_ref_embs[valid_obj_indexes][:sample_obj_lens]


                # rule1: human features sim
                sort_indexes = sim.sort(descending=True)[1]
                valid_indexes = sort_indexes[torch.where(sort_indexes != k)[0]]

                sample_lens = min(len(valid_indexes),topk)

                sim_hum_feats = new_hum[valid_indexes][:sample_lens]
                sim_hum_area = new_hum_area[valid_indexes][:sample_lens]

                # rule2: object 
                sort_obj_indexes = sim_obj[k].sort(descending=True)[1]
                current_obj = object_indexes_pool[k]
                valid_obj_indexes = sort_obj_indexes[torch.where(sort_obj_indexes != current_obj)[0]]

                sample_obj_lens = min(len(valid_obj_indexes),topk)

                sim_obj_feats = object_ref_embs[valid_obj_indexes][:sample_obj_lens]
                sim_obj_area = ref_obj_area[valid_obj_indexes][:sample_obj_lens]

                # rule3 topk 
                hum_obj_ratio_one = hum_obj_ratio[k]
                hum_area_one = hum_area[k]
                obj_area_one = obj_area[k]
                # if i == 8: pdb.set_trace()
                for h_i, h_feat in enumerate(sim_hum_feats):
                    h_area = sim_hum_area[h_i]
                    for o_i, o_feat in enumerate(sim_obj_feats):
                        o_area = sim_obj_area[o_i]
                        # ratio = h_area/o_area /hum_obj_ratio_one
                        if torch.abs(h_area/hum_area_one -1)<0.1 and torch.abs(o_area/obj_area_one -1)<0.1:
                            save_one_hum.append(h_feat)
                            save_one_obj.append(o_feat)
                            new_count += 1

                # sample_lens = min(len(valid_indexes),5)
                # sample_hum_feats = new_hum[valid_indexes][:sample_lens]


                # final_hum_feats = sample_hum_feats.unsqueeze(1).repeat(1,sample_obj_lens,1).view(-1,512)
                # final_obj_feats = sample_obj_feats.unsqueeze(0).repeat(sample_lens,1,1).view(-1,512)
                # save_one_hum.append(final_hum_feats)
                # save_one_obj.append(final_obj_feats)
        else:
            pass
        print(i)
        # pdb.set_trace()
        all_count.append(new_count)
        if len(save_one_hum) != 0:
            final_hum = torch.cat([hum_emb,torch.stack(save_one_hum)])
            final_obj = torch.cat([obj_emb,torch.stack(save_one_obj)])
        else:
            final_hum = hum_emb
            final_obj = obj_emb
        gt_sample_lens = lens
        new_dict = {}
        new_dict['final_hum'] = final_hum
        new_dict['final_obj'] = final_obj
        new_dict['gt_sample_lens'] = gt_sample_lens
        save_all_list.append(new_dict)
    #     topk_lens = min(6, len(sim_))
    #     topk_sample = sim_.topk(topk_lens,dim=-1)[1][:lens] 

    pdb.set_trace()    

    with open("same_HOI_extention_all_objects_3rule_new_two_100samples.p","wb") as f:
        pickle.dump(save_all_list,f)

        
else:
    def draw_pic(embeddings, original_lens, save_dir,title):
        # pkl_file = '../embs_ye.p'
        # with open(pkl_file, 'rb') as f:
        #     new_embeddings = pickle.load(f)
        new_embeddings = embeddings/embeddings.norm(dim=-1,keepdim=True)
        # title = 'load airplane'
        title = title

        # origin_idx = torch.arange(0, new_embeddings.shape[0], 2)

        origin_idx = torch.arange(0,original_lens)

        tsne = TSNE(n_components=2, perplexity=30, n_iter=3000)
        tsne.fit_transform(new_embeddings)
        embeddings_tsne = tsne.embedding_
        

        colors = cm.rainbow(np.linspace(0, 1, 2))
        plt.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], s=1, marker='o', label='augmented_{}'.format(new_embeddings.shape[0]-len(origin_idx)), color=colors[0])
        plt.scatter(embeddings_tsne[origin_idx, 0], embeddings_tsne[origin_idx, 1], s=1, marker='o', label='real_{}'.format(len(origin_idx)), color=colors[1])

        plt.title(title)
        plt.legend(loc='best')
        plt.savefig(save_dir)
        plt.clf()
        plt.close()
        return 

    anno_file = pickle.load(open('same_verb_extention_all_objects_3rule_new_five.p','rb'))
    lens = len(anno_file)
    for i, anno in enumerate(anno_file):
        # pdb.set_trace()
        gt_sample_lens = anno['gt_sample_lens'] 
        final_hum = anno['final_hum']
        final_obj = anno['final_obj']
        
        save_embs = torch.cat([final_hum,final_obj],dim=-1)
        if len(save_embs) < 2:
            print(i)
            continue
        
        if (i in unseen_rare_first and len(save_embs)>=2) or gt_sample_lens<32 or i==0:
        # if i ==0:
            save_dir = 'save_visual_new/{}_{}.png'.format('_'.join(hico_verb_object_list[i]),gt_sample_lens)
            draw_pic(save_embs, gt_sample_lens, save_dir,' '.join(hico_verb_object_list[i]))

        print("current run: {}".format(i))
    print("finish!!!")