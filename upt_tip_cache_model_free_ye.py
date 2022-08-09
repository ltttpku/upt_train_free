"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from __future__ import annotations
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits
from interaction_head import InteractionHead

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from hico_text_label import hico_text_label, hico_obj_text_label
sys.path.append('detr')
# print(sys.path)
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list

import pdb
# from CLIP_models import CLIP_ResNet, tokenize
from CLIP_models_adapter_prior import CLIP_ResNet, tokenize
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_tokenizer = _Tokenizer()
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        detector: nn.Module,
        postprocessor: nn.Module,
        clip_head: nn.Module,
        clip_pretrained: str,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None, 
        topk: int = 250,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = clip_head

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class


        device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model, _ = clip.load("ViT-B/16", device=device)


        self.class_nums = 600
        use_templates = False
        if self.class_nums==117 :text_inputs = torch.cat([clip.tokenize(verb) for verb in hico_verbs]) # hico_verbs 'action is ' +
        elif self.class_nums==600 and use_templates==False:
            text_inputs = torch.cat([clip.tokenize(hico_text_label[id]) for id in hico_text_label.keys()])
        elif self.class_nums==600 and use_templates==True:
            text_inputs = self.get_multi_prompts(hico_text_label)
            bs_t, nums, c = text_inputs.shape
            text_inputs = text_inputs.view(-1, c)
        with torch.no_grad():
            text_embedding = self.clip_model.encode_text(text_inputs.to(device))
        if use_templates:
            text_embedding = text_embedding.view(bs_t, nums, -1).mean(0)

        # use object embedding 
        hico_triplet_labels = list(hico_text_label.keys())
        hoi_obj_list = []
        for hoi_pair in hico_triplet_labels:
            hoi_obj_list.append(hoi_pair[1])
        obj_text_inputs = torch.cat([clip.tokenize(obj_text[1]) for obj_text in hico_obj_text_label])

        with torch.no_grad():
            obj_text_embedding = self.clip_model.encode_text(obj_text_inputs.to(device))[hoi_obj_list,:]

        obj_hoi_embedding = torch.cat([obj_text_embedding,text_embedding],dim=-1)
        self.obj_hoi_embedding = obj_hoi_embedding/obj_hoi_embedding.norm(dim=-1, keepdim=True)

        '''
        To Do
        '''
        self.text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 600*512
        # self.text_sim = self.text_embedding @ self.text_embedding.t()  # similarity of the text embeddings

        self.logit_scale = nn.Parameter(torch.ones([]) * 10)

        self.HOI_IDX_TO_ACT_IDX = [
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

        self.HOI_IDX_TO_OBJ_IDX = [
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

        ## to do
        self.dicts = {}
        self.feature = 'hum_obj' # union, obj_uni, hum_uni, hum_obj_uni, cahce model 
        ## arguments
        self._load_features = True
        self.branch = kwargs['branch'] ## F_cluster, F_cluster_semantic_label, F_vis, F_vis_semantic_label
                                  ## F_cluster_F_vis_semantic_label， text_only, vis+text+cluster
        self.temperature = 1
        self.topk = topk
        num_shot = kwargs['num_shot']
        iou_rank = False
        self.use_kmeans = False # # priority: kmeans>iou_rank
        use_mean_feat = False
        self.use_outliers = kwargs['use_outliers']
        self.use_less_confident = kwargs['use_less_confident'] ## priority: use_less_confident > use_outliers > kmeans > iou_rank

        self.preconcat = True
        self._global_norm = False
        # file1 = 'union_embeddings_cachemodel_clipcrops.p'
        # file1 = 'union_embeddings_cachemodel_crop_padding_zeros.p'
        file1 = 'union_embeddings_cachemodel_crop_padding_zeros_vitb16.p'
        print('[INFO]: in _ye.py')
        print('>>> branch:', self.branch)
        print('>>> feature_type:', self.feature, 'topk:', self.topk, \
                ' use_kmeans:',self.use_kmeans, ' iou_rank:',iou_rank, \
                ' use_mean_feat:',use_mean_feat, ' softmax_temperature:',self.temperature)
        print('[INFO]: use_outliers:', self.use_outliers, )
        print('[INFO]: num_shot:', num_shot, 'preconcat:', self.preconcat, 'global_norm:', self._global_norm)
        # save_file2 = 'save_sample_indexes_{}_{}.p'.format(self.class_nums,num_shot)
        save_file2 = None
        self.hois_cooc = torch.load('one_hots.pt')

        self.cache_models, self.one_hots, self.sample_lens = self.load_cache_model(file1=file1,file2=save_file2, feature=self.feature,class_nums=self.class_nums, num_shot=num_shot,iou_rank=iou_rank,use_mean_feat=use_mean_feat)
        # pdb.set_trace()
        # self.verb_cache_models, self.verb_one_hots, self.verb_sample_lens = self.load_cache_model(file1=file1,file2=save_file2, feature=self.feature,class_nums=117, num_shot=num_shot,iou_rank=False,use_mean_feat=False)
        if self._global_norm:
            self.cache_models /= self.cache_models.norm(dim=-1, keepdim=True)

        dicts= pickle.load(open('test_kmeans.p','rb'))

        self.cache_models = self.cache_models.cuda().float()  
        # self.verb_cache_models = self.verb_cache_models.cuda().float()      
        print('cachemodel.shape:', self.cache_models.shape)

        self.use_type = 'crop'
        self.one_hots = self.one_hots.cuda().float()
        # self.verb_one_hots = self.verb_one_hots.cuda().float()
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)
        self.sample_lens = torch.as_tensor(self.sample_lens).cuda()
        # self.verb_sample_lens = torch.as_tensor(self.verb_sample_lens).cuda()

        ###  load cluster (F_cluster)
        anno_cluster = pickle.load(open('clusters_80_117_padding_zeros.p','rb'))
        self.human_clusters = anno_cluster['cluster_centers_hum'].cuda() # C x 512
        self.object_clusters = anno_cluster['cluster_centers_obj'].cuda() # C x 512
        self.hoi_clusters = anno_cluster['hoi_scores'].cuda() # 600 x C
        # self.hoi_clusters = self.hoi_clusters/self.hoi_clusters.norm(dim=-1,keepdim=True)
        
        self.finetune_adapter = False
        if self.finetune_adapter:
            self.adapter = nn.Linear(self.cache_models.shape[1], self.cache_models.shape[0], bias=True)
            self.adapter.weight = nn.Parameter(self.cache_models.float())
            self.adapter.bias = nn.Parameter(-torch.ones(self.cache_models.shape[0]))

        self.evaluate_type = 'detr' # gt detr
        self.post_process = kwargs['post_process']
        print('[INFO]: post_process:', self.post_process)
        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]

        
        self.obj_to_no_interaction = torch.as_tensor([169,  23,  75, 159,   9,  64, 193, 575,  45, 566, 329, 505, 417, 246,
         30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
         91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        

        cumsum_sample_lens = torch.cumsum(self.sample_lens, dim=-1) ## 32, 32, 13, 27, 32, 
        self.Lp_sample_lens = torch.zeros(117).cuda()
        self.Lp_one_hots = torch.zeros(cumsum_sample_lens[-1], 117).cuda().float()
        for i in range(600):
            if i == 0:
                self.Lp_one_hots[0:cumsum_sample_lens[i], self.HOI_IDX_TO_ACT_IDX[i]] = 1
                self.Lp_sample_lens[self.HOI_IDX_TO_ACT_IDX[i]] += cumsum_sample_lens[i]
            else:
                self.Lp_one_hots[cumsum_sample_lens[i-1]:cumsum_sample_lens[i], self.HOI_IDX_TO_ACT_IDX[i]] = 1 
                self.Lp_sample_lens[self.HOI_IDX_TO_ACT_IDX[i]] += (cumsum_sample_lens[i] - cumsum_sample_lens[i-1])
        # self.Lp_sample_lens = self.Lp_sample_lens.cuda()
        # self.Lp_one_hots = self.Lp_one_hots.cuda().float()

        # calculate invalid pairs acc
        self.r1_nopair = 0
        self.count_nopair = 0

        self.r1_pair = 0
        self.count_pair = 0

    def old_prompt2new_prompt(self, prompt):
        '''
        param prompt: str, e.g. 'a photo of a person holding a bicycle'
        '''
        # pdb.set_trace()
        new_prefix = 'interaction between human and an object, '
        ## remove old prefix
        prompt = ' '.join(prompt.split()[3:])
        return new_prefix + prompt

    def get_multi_prompts(self, hico_labels):   ## xx
        templates = ['itap of {}', '“a bad photo of {}', 'a photo of {}', 'there is {} in the video game', 'art of {}', 'the picture describes {}']
        hico_texts = [hico_text_label[id].split(' ')[3:] for id in hico_text_label.keys()]
        all_texts_input = []
        for temp in templates:
            texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
            all_texts_input.append(texts_input)
        all_texts_input = torch.stack(all_texts_input,dim=0)
        return all_texts_input
    
    def get_attention_feature(self, query_feat, human_feat, object_feat, ftype='patch'):  ## xxx
        device = torch.device('cuda')
        
        human_feat = human_feat.flatten(2).to(device)
        object_feat = object_feat.flatten(2).to(device)
        key_feat = torch.cat([human_feat,object_feat],dim=-1)

        query_feat = query_feat.flatten(2).transpose(1,2).to(device)
        # pdb.set_trace()
        global_feat = query_feat.mean(1)
        # key_feat = key_feat/key_feat.norm(dim=1, keepdim=True)
        # query_feat = query_feat/query_feat.norm(dim=-1, keepdim=True)
        weight_matrix = torch.bmm(query_feat, key_feat)
        weight_matrix = weight_matrix.float().softmax(-1)
        weight_query_feat = torch.bmm(weight_matrix, key_feat.transpose(1, 2).float()).mean(1)
        query_feat = weight_query_feat.half()
        return query_feat.cpu()

    def naive_kmeans(self, feats, K):
        '''
        feats: num x 512 (num > K)
        return: K x 512
        '''
        # pdb.set_trace()
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(feats)
        cluster_centers = kmeans.cluster_centers_
        return cluster_centers
    
    def search_best_k_for_kmeans(self, data, ):
        """
        Search the best k for kmeans
        """
        k_range = range(5, 33, 1)
        k_scores = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(data)
            k_scores.append(silhouette_score(data, kmeans.labels_))
        return k_range[np.argmax(k_scores)]

    def get_kmeans_cluster_centers(self, data,):
        """
        Get kmeans cluster
        """
        k = self.search_best_k_for_kmeans(data)
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        return kmeans.cluster_centers_

    def intra_swapping(self, feats, n_components=2):
        num, dim = feats.shape
        assert dim == 1024
        # pdb.set_trace()
        if n_components == 2:
            part1 = feats[:, :512]
            part2 = feats[:, 512:]
            part1 = part1.repeat(num, 1)
            part2 = part2.repeat(1, num).reshape(-1, 512)
            feats = torch.cat((part1, part2), dim=-1)
            return feats
        else:
            raise NotImplementedError

    def select_outliers(self, feats, K, outlier=True, method='default', text_embedding=None, origin_idx=None, divisor=2):
        '''
        feats: num x 512 (num >= K), tensor, dtype=float32
        return: K x 512
        '''
        if method == 'fps':
            return self.fps(feats, K)
        elif method == 'text':
            return self.use_less_confident(feats, K, text_embedding)  
        # pdb.set_trace()
        # indices = torch.randperm(feats.shape[0])[:10000]
        # feats = feats[indices]

        feats = feats.cuda().float()
        num_of_neighbours = feats.shape[0] // divisor
        ## calculate the distance between each pair of features
        dis_matrix = feats @ feats.t()
        
        if num_of_neighbours > 0:
            ## select the k smallest num for every row of the distance matrix
            dis_matrix = torch.sort(dis_matrix, dim=1)[0]
            ## sum the k smallest nums for every row
            dis_vector = (dis_matrix[:,:num_of_neighbours].sum(1)) / num_of_neighbours
        else: ## neighours==all other vectors
            dis_vector = dis_matrix.mean(dim=1)
        
        # pdb.set_trace()
        if origin_idx != None:
            dis_vector[origin_idx] += 0.1
        ## select topk largest and topk smallest distance
        # pdb.set_trace()
        # idx = torch.cat((torch.argsort(dis_vector, descending=outlier)[:K - K //10], torch.argsort(dis_vector, descending=outlier)[-(K//10):]), dim=0)
        # return feats[idx]
        topk_idx = torch.argsort(dis_vector, descending=outlier)[:K]
        return topk_idx
        # topk_feats = feats[topk_idx]
        # return topk_feats.cpu()
    
    def fps(self, points, n_samples):
        """
        points: [N, 3] array containing the whole point cloud
        n_samples: samples you want in the sampled point cloud typically << N 
        """
        # pdb.set_trace()
        def top1_outlier_idx(feats):
            feats = torch.from_numpy(feats)
            num_of_neighbours = feats.shape[0] // 3
            dis_matrix = feats @ feats.t()
    
            if num_of_neighbours > 0:
                dis_matrix = torch.sort(dis_matrix, dim=1)[0]
                dis_vector = dis_matrix[:,:num_of_neighbours].sum(1)
            else: ## neighours==all other vectors
                dis_vector = dis_matrix.mean(dim=1)
            topk_idx = torch.argsort(dis_vector, descending=True)[:1]
            return topk_idx

        points = np.array(points.cpu())
        # Represent the points by their indices in points
        points_left = np.arange(len(points)) # [P]
        # Initialise an array for the sampled indices
        sample_inds = np.zeros(n_samples, dtype='int') # [S]
        # Initialise distances to inf
        dists = np.ones_like(points_left) * float('inf') # [P]
        # Select a point from points by its index, save it
        selected = top1_outlier_idx(points)
        # pdb.set_trace()
        sample_inds[0] = points_left[selected]

        # Delete selected 
        points_left = np.delete(points_left, selected) # [P - 1]

        # Iteratively select points for a maximum of n_samples
        for i in range(1, n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i-1]
            dist_to_last_added_point = (
                (points[last_added] - points[points_left])**2).sum(-1) # [P - i]
            
            dist_lst = [(
                (points[idx] - points[points_left])**2).sum(-1) for idx in sample_inds[:i]] # [P - i]
            dist_to_added_points = np.array(dist_lst)
            dist_to_added_points = dist_to_added_points.min(axis=0) # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_added_points, 
                                            dists[points_left]) # [P - i]
            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]
            # Update points_left
            points_left = np.delete(points_left, selected)
        return torch.as_tensor(sample_inds)

    
    def select_less_confident(self, feats, K, text_feat, less_confident=True):
        '''
        params:
            feats: num x 512 (num >= K), tensor, dtype=float32
            text_feat: 512, tensor, dtype=float32
        return: K x 512
        '''
        ## calculate the distance between each feat and text_feat
        dis_vector = feats @ text_feat.t()
        ## select topk largest distance
        topk_idx = torch.argsort(dis_vector, descending=less_confident)[:K]
        return topk_idx
        

    def load_cache_model_lt(self, file1, K_shot=32):
        annotation = pickle.load(open(file1,'rb'))
        categories = 600
        union_embeddings = [[] for i in range(categories)]
        obj_embeddings = [[] for i in range(categories)]
        hum_embeddings = [[] for i in range(categories)]
        filenames = list(annotation.keys())

        each_filenames = [[] for i in range(categories)]
        sample_indexes = [[] for i in range(categories)]
        for file_n in filenames:
            anno = annotation[file_n]
            if categories == 117: verbs = anno['verbs']
            else: verbs = anno['hois']
            
            union_features = anno['union_features']
            object_features = anno['object_features']
            huamn_features = anno['huamn_features']
            ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
            
            if len(verbs) == 0:
                print(file_n)
            for i, v in enumerate(verbs):
                union_embeddings[v].append(union_features[i] / np.linalg.norm(union_features[i]))
                obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i]))
                hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))
                each_filenames[v].append(file_n)
                sample_indexes[v].append(i)
                # add iou
                # verbs_iou[v].append(ious[i])
        cluster_center_lst = []
        cluster_num_lst = []
        indexes = np.arange(len(union_embeddings))
        
        for i, hum_emb, obj_emb, embeddings in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings)):
            hum_emb, obj_emb, embeddings = torch.as_tensor(hum_emb), torch.as_tensor(obj_emb), torch.as_tensor(embeddings)
            range_lens = np.arange(len(embeddings))
            feats = torch.cat([hum_emb, obj_emb, embeddings], dim=-1)
            if len(range_lens) > K_shot:
                ## assert using hum+obj+union
                cluster_centers  = self.get_kmeans_cluster_centers(feats)
                cluster_center_lst.append(torch.from_numpy(cluster_centers))
                cluster_num_lst.append(cluster_centers.shape[0])
            else:
                cluster_center_lst.append(feats)
                cluster_num_lst.append(feats.shape[0])
        # pdb.set_trace()
        cache_model = torch.cat(cluster_center_lst, dim=0)
        sample_lens = torch.as_tensor(cluster_num_lst)
        cumsum_sample_lens = torch.cumsum(sample_lens, dim=-1)
        one_hots = torch.zeros(cumsum_sample_lens[-1], 600)
        for i in range(600):
            if i == 0:
                one_hots[0:cumsum_sample_lens[i], i] = 1
            else:
                one_hots[cumsum_sample_lens[i-1]:cumsum_sample_lens[i], i] = 1 
        return cache_model, one_hots, cluster_num_lst
        


    def load_cache_model(self,file1, file2=None, category='verb', feature='union',class_nums=117, num_shot=10, iou_rank=False, use_mean_feat=False):  ## √
        '''
        To Do
        '''
        # pdb.set_trace()
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
        for file_n in filenames:
            anno = annotation[file_n]
            if categories == 117: verbs = anno['verbs']
            else: verbs = anno['hois']
            
            union_features = anno['union_features']
            object_features = anno['object_features']
            # pdb.set_trace()
            huamn_features = anno['huamn_features']
            ious = torch.diag(box_iou(torch.as_tensor(anno['boxes_h']), torch.as_tensor(anno['boxes_o'])))
            
            if len(verbs) == 0:
                print(file_n)
            for i, v in enumerate(verbs):
                # pdb.set_trace()
                union_embeddings[v].append(union_features[i] / np.linalg.norm(union_features[i]))
                obj_embeddings[v].append(object_features[i] / np.linalg.norm(object_features[i]))
                hum_embeddings[v].append(huamn_features[i] / np.linalg.norm(huamn_features[i]))
                each_filenames[v].append(file_n)
                sample_indexes[v].append(i)
                # add iou
                verbs_iou[v].append(ious[i])
                
        all_lens = torch.as_tensor([len(u) for u in union_embeddings])
        K_shot = num_shot
        if use_mean_feat:
            all_lens_sample = torch.ones(categories)
            all_lens_sample = torch.cumsum(all_lens_sample,dim=-1).long()
        else:
            K_sample_lens = all_lens>K_shot
            all_lens_sample = torch.ones(categories) * K_shot
            index_lessK = torch.where(K_sample_lens==False)[0]
            # pdb.set_trace()
            for index in index_lessK:
                all_lens_sample[index] = len(union_embeddings[index])
            all_lens_sample = torch.cumsum(all_lens_sample,dim=-1).long()
        if file2 is not None:
            save_sample_index = pickle.load(open(file2, 'rb'))['sample_index']
        else:
            save_sample_index = []
        save_filenames = []
        save_sample_indexes = []
        # pdb.set_trace()
        all_if_origin = []

        if feature == 'union' :
            cache_models = torch.zeros((all_lens_sample[-1], 512),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings in zip(indexes, hum_embeddings, obj_embeddings, union_embeddings):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) > K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    elif iou_rank==True:
                        ious = torch.as_tensor(verbs_iou[i])
                        _, iou_inds = ious.sort()
                        sample_ind = np.arange(0, len(ious), len(ious)/lens, dtype=np.int)[:lens]
                        sample_index = iou_inds[sample_ind]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot,replace=False)
                    sample_embeddings = np.array(embeddings)[sample_index]
                    sample_obj_embeddings =  np.array(obj_emb)[sample_index]
                    sample_hum_embeddings =  np.array(hum_emb)[sample_index]
                    ## kmeans 
                    ## embeddings, obj_emb, hum_emb: lst of tensors, same shape
                    ## sample_embeddings: K_shot x 512
                    # pdb.set_trace()
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = np.array(embeddings)
                    sample_obj_embeddings = np.array(obj_emb)
                    sample_hum_embeddings = np.array(hum_emb)
                
                if i==0:
                    cache_models[:all_lens_sample[i],:] = torch.as_tensor(sample_embeddings)
                    
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot
                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:] = torch.as_tensor(sample_embeddings)

                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot

                each_lens.append(lens)
        elif feature == 'hum_obj':
            cache_models = torch.zeros((all_lens_sample[-1], 512*2),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings in tqdm(zip(indexes, hum_embeddings, obj_embeddings, union_embeddings)):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) > K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    elif iou_rank==True:
                        ious = torch.as_tensor(verbs_iou[i])
                        _, iou_inds = ious.sort()
                        sample_ind = np.arange(0, len(ious), len(ious)/lens, dtype=np.int)[:lens]
                        sample_index = iou_inds[sample_ind]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot,replace=False)
                    
                    sample_obj_embeddings =  torch.as_tensor(np.array(obj_emb)[sample_index])
                    sample_hum_embeddings =  torch.as_tensor(np.array(hum_emb)[sample_index])
                    
                    if self.use_kmeans:
                        sample_obj_embeddings = torch.from_numpy(self.naive_kmeans(np.array(obj_emb), K=K_shot))
                        sample_hum_embeddings = torch.from_numpy(self.naive_kmeans(np.array(hum_emb), K=K_shot))
                    # pdb.set_trace()
                    if self.use_outliers:
                        obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                        hum_emb =  torch.as_tensor(np.array(hum_emb)).float()
                        if self.preconcat:
                            new_embeddings = torch.cat([ obj_emb, hum_emb], dim=-1) ## concat before outlier, true
                            new_embeddings = new_embeddings.cuda().float()
                            
                            if self._global_norm:
                                new_embeddings /= new_embeddings.norm(dim=-1, keepdim=True)
                            if new_embeddings.shape[0] < 100:
                                origin_embeddings = new_embeddings.clone().detach() ## 60 x 1024
                                new_embeddings = self.intra_swapping(new_embeddings) ## 3600 x 1024
                                origin_idx = torch.arange(0, new_embeddings.shape[0], obj_emb.shape[0]).cuda()
                                # pdb.set_trace()
                                dis_vector = (new_embeddings @ origin_embeddings.t()).mean(dim=-1)
                                topk_indices = torch.argsort(dis_vector, descending=True)[:new_embeddings.shape[0]//2]
                                new_embeddings = new_embeddings[topk_indices]

                                # if self.alpha > 0.5: ## tend to select 'real'
                                #     topk_idx = self.select_outliers(new_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i], origin_idx=origin_idx)
                                # else:
                                # pdb.set_trace()
                                topk_idx = self.select_outliers(new_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])
                                # topk_idx = torch.randperm(new_embeddings.shape[0])[:K_shot].cuda() ###### random choice 1
                                if_origin = [topk_idx[i] in origin_idx for i in range(K_shot)]
                                # print(sum(if_origin) / K_shot)
                                all_if_origin.extend(if_origin)
                            else:
                                topk_idx = self.select_outliers(new_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])
                                # topk_idx = torch.randperm(new_embeddings.shape[0])[:K_shot] ###### random choice 2
                            sample_obj_embeddings = new_embeddings[topk_idx, :512]
                            sample_hum_embeddings = new_embeddings[topk_idx, 512:]
                        else:
                            # pdb.set_trace()
                            ## inter-class
                            # inter_hum_indices = (torch.as_tensor(self.HOI_IDX_TO_ACT_IDX) == self.HOI_IDX_TO_ACT_IDX[i]).nonzero()
                            # inter_hum_embeddings = [torch.as_tensor(hum_embeddings[i]) for i in inter_hum_indices]
                            # inter_hum_embeddings = torch.cat((inter_hum_embeddings), dim=0)

                            # inter_obj_indices = (torch.as_tensor(self.HOI_IDX_TO_OBJ_IDX) == self.HOI_IDX_TO_OBJ_IDX[i]).nonzero()
                            # inter_obj_embeddings = [torch.as_tensor(obj_embeddings[i]) for i in inter_obj_indices]
                            # inter_obj_embeddings = torch.cat((inter_obj_embeddings), dim=0)

                            # sample_obj_embeddings = inter_obj_embeddings[self.select_outliers(inter_obj_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                            # sample_hum_embeddings = inter_hum_embeddings[self.select_outliers(inter_hum_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                    
                            sample_obj_embeddings = obj_emb[self.select_outliers(obj_emb, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                            sample_hum_embeddings = hum_emb[self.select_outliers(hum_emb, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                    
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = torch.as_tensor(np.array(embeddings))
                    sample_obj_embeddings = torch.as_tensor(np.array(obj_emb))
                    sample_hum_embeddings = torch.as_tensor(np.array(hum_emb))

                lens = len(sample_obj_embeddings)
                if i==0:
                    cache_models[:all_lens_sample[i],:512] = sample_hum_embeddings
                    cache_models[:all_lens_sample[i],512:1024] = sample_obj_embeddings
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot

                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:512] = sample_hum_embeddings
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],512:1024] = sample_obj_embeddings
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot

                each_lens.append(lens)

        elif feature == 'hum_obj_uni':
            cache_models = torch.zeros((all_lens_sample[-1], 512*3),dtype=torch.float16)
            one_hots = torch.zeros((all_lens_sample[-1], categories),dtype=torch.float16)
            each_lens = []
            indexes = np.arange(len(union_embeddings))
            for i, hum_emb, obj_emb, embeddings in zip(indexes, hum_embeddings, obj_embeddings, union_embeddings):
                range_lens = np.arange(len(embeddings))
                if len(range_lens) > K_shot:
                    lens = K_shot
                    if file2 is not None:
                        sample_index = save_sample_index[i]
                    elif iou_rank==True:
                        # pdb.set_trace()
                        ious = torch.as_tensor(verbs_iou[i])
                        _, iou_inds = ious.sort()
                        sample_ind = np.arange(0, len(ious), len(ious)/lens, dtype=np.int)[:lens]
                        sample_index = iou_inds[sample_ind]
                    else:
                        sample_index = np.random.choice(range_lens,K_shot,replace=False)
                    
                    sample_embeddings = torch.as_tensor(np.array(embeddings)[sample_index])
                    sample_obj_embeddings =  torch.as_tensor(np.array(obj_emb)[sample_index])
                    sample_hum_embeddings =  torch.as_tensor(np.array(hum_emb)[sample_index])
                    # pdb.set_trace()
                    if self.use_kmeans:
                        sample_embeddings = torch.from_numpy(self.naive_kmeans(np.array(embeddings), K=K_shot))
                        sample_obj_embeddings = torch.from_numpy(self.naive_kmeans(np.array(obj_emb), K=K_shot))
                        sample_hum_embeddings = torch.from_numpy(self.naive_kmeans(np.array(hum_emb), K=K_shot))
                    # pdb.set_trace()
                    if self.use_outliers:
                        embeddings = torch.as_tensor(np.array(embeddings)).float()
                        obj_emb = torch.as_tensor(np.array(obj_emb)).float()
                        hum_emb =  torch.as_tensor(np.array(hum_emb)).float()
                        if self.preconcat:
                            new_embeddings = torch.cat([embeddings, obj_emb, hum_emb], dim=-1)
                            if self._global_norm:
                                new_embeddings /= new_embeddings.norm(dim=-1, keepdim=True)
                            topk_idx = self.select_outliers(new_embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])
                            sample_embeddings = embeddings[topk_idx]
                            sample_obj_embeddings = obj_emb[topk_idx]
                            sample_hum_embeddings = hum_emb[topk_idx]
                        else:
                            # pdb.set_trace()
                            sample_embeddings = embeddings[self.select_outliers(embeddings, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                            sample_obj_embeddings = obj_emb[self.select_outliers(obj_emb, K=K_shot, method='default', text_embedding=self.text_embedding[i])]
                            sample_hum_embeddings = hum_emb[self.select_outliers(hum_emb, K=K_shot, method='default', text_embedding=self.text_embedding[i])]

                            # sample_embeddings = sample_embeddings[:4, :].repeat(K_shot//4, 1)
                            # sample_obj_embeddings = sample_obj_embeddings[:8, :].repeat(K_shot//8, 1)
                else:
                    lens = len(embeddings)
                    sample_index = np.arange(lens)
                    sample_embeddings = torch.as_tensor(np.array(embeddings))
                    sample_obj_embeddings = torch.as_tensor(np.array(obj_emb))
                    sample_hum_embeddings = torch.as_tensor(np.array(hum_emb))
                lens = len(sample_embeddings)
                if i==0:
                    cache_models[:all_lens_sample[i],:512] = sample_hum_embeddings
                    cache_models[:all_lens_sample[i],512:1024] = sample_obj_embeddings
                    cache_models[:all_lens_sample[i],1024:] = sample_embeddings
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[:all_lens_sample[i],:] = one_hot

                else:
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],:512] = sample_hum_embeddings
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],512:1024] = sample_obj_embeddings
                    cache_models[all_lens_sample[i-1]:all_lens_sample[i],1024:] = sample_embeddings
                    one_hot = torch.zeros((lens,categories),dtype=torch.long)
                    one_hot[:,i] = 1
                    one_hots[all_lens_sample[i-1]:all_lens_sample[i],:] = one_hot

                each_lens.append(lens)
        if len(all_if_origin) > 0:
            print('>>> real / (real+augmented) =', sum(all_if_origin) / len(all_if_origin))
        return cache_models,one_hots, each_lens
        

    def get_clip_feature(self,image):  ## xxx
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # pdb.set_trace()
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        # x = self.clip_model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        return local_feat
    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:  ### √
        # pdb.set_trace()
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        # pdb.set_trace()
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        # p = 1.0
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
        
    def compute_text_embeddings(self):  ### xxx
        
        text_embeddings = self.clip_head.text_encoder(self.texts)
        return text_embeddings
    def compute_roi_embeddings_targets(self, features: OrderedDict, image_shapes: Tensor, targets_region_props: List[dict], return_type='crop'): ### √

        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        all_logits = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_boxes = []
        for i, targets in enumerate(targets_region_props):
            # pdb.set_trace()
            local_features = features[i]
            gt_bx_h = (box_ops.box_cxcywh_to_xyxy(targets['boxes_h']) * scale_fct[i][None,:])
            gt_bx_o = (box_ops.box_cxcywh_to_xyxy(targets['boxes_o']) * scale_fct[i][None,:])
            verbs = targets['labels']
            hois = targets['hoi']
            filename = targets['filename']
            objects_label = targets['object']
            lt = torch.min(gt_bx_h[..., :2], gt_bx_o[..., :2]) # left point
            rb = torch.max(gt_bx_h[..., 2:], gt_bx_o[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            if return_type == 'roi':
                union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
                huamn_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_h],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
                object_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[gt_bx_o],output_size=(7, 7),spatial_scale=1 / 32.0,aligned=True).flatten(2).mean(-1)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
                logits = union_features @ self.text_embedding.t()
                logits_cache = ((union_features @ self.cache_models.t()) @ self.one_hots) / self.sample_lens
                logits = logits + logits_cache
            elif return_type == 'crop':  # to do -> cache model 
                lens = local_features.shape[0]
                feat1 = local_features[:lens//3,:] # human features
                feat2 = local_features[lens//3:lens//3*2,:] # object features
                feat3 = local_features[lens//3*2:,:]  # union features
                # pdb.set_trace()
                feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
                feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
                feat3 = feat3 / feat3.norm(dim=-1, keepdim=True)
                
                # pdb.set_trace()
                score_h = feat1 @ self.human_clusters.t()
                score_h /= score_h.norm(dim=-1, keepdim=True)
                score_o = feat2 @ self.object_clusters.t()
                score_o /= score_o.norm(dim=-1, keepdim=True)
                # pdb.set_trace()
                score = torch.cat((score_o, score_h), dim=1) ## dim: C1 + C2

                f_cluster = score

                feat_human_obj_union = torch.cat((feat1, feat2,feat3), dim=-1)
                feat_obj_union = torch.cat((feat2,feat3), dim=-1)
                feat_hum_union = torch.cat((feat1,feat3), dim=-1)

                if self.feature == 'hum_uni':
                    pass
                else:
                    print(f"[ERROR]: feature_type {self.feature} not implemented yet")

                # pdb.set_trace()
                if self.branch == 'F_cluster':
                    pass
            else:
                print('please input the correct return type: roi or crop')
                sys.exit()

            all_boxes.append(torch.cat([gt_bx_h,gt_bx_o],dim=0))
            keep = torch.arange(len(gt_bx_h)*2).to(local_features.device)
            boxes_h_collated.append(keep[:len(gt_bx_h)])
            boxes_o_collated.append(keep[len(gt_bx_h):])
            object_class_collated.append(objects_label)
            scores = torch.ones(len(keep)).to(local_features.device)
            
            prior_collated.append(self.compute_prior_scores(
                keep[:len(gt_bx_h)], keep[:len(gt_bx_o)], scores, objects_label)
            )
            all_logits.append(logits.float())
        return all_logits,prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated ,all_boxes


    def compute_roi_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict]): ### xx
        pass

    def compute_crop_embeddings(self, features: OrderedDict, image_shapes: Tensor, region_props: List[dict], targets_region_props: List[dict], return_type='crop'): ### √
        img_h, img_w = image_shapes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        all_logits = []
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        all_boxes = []
        device = features.device
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            # local_features = features[:,b_idx,:]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']
            # add mask
            masks = props['mask']
            is_human = labels == self.human_idx
            
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]; unary_tokens = unary_tokens[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()


            # extract single roi features
            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)
            
            # just get the 匹配的
            
            # gt_bx_h = self.recover_boxes(targets_region_props[b_idx]['boxes_h'], targets_region_props[b_idx]['size'])
            # gt_bx_o = self.recover_boxes(targets_region_props[b_idx]['boxes_o'], targets_region_props[b_idx]['size'])
            
            # x, y = torch.nonzero(torch.min(
            #     box_iou(sub_boxes, gt_bx_h),
            #     box_iou(obj_boxes, gt_bx_o)
            #     ) >= self.fg_iou_thresh).unbind(1)
            # # 
            # if len(x) == 0: 
            #     print(x,y)
            #     self.count += 1
            # x_keep = x_keep[x]
            # y_keep = y_keep[x]

            gt_bx_h = self.recover_boxes(targets_region_props[b_idx]['boxes_h'], targets_region_props[b_idx]['size'])
            gt_bx_o = self.recover_boxes(targets_region_props[b_idx]['boxes_o'], targets_region_props[b_idx]['size'])
            
            x, y = torch.nonzero(torch.min(
                box_iou(sub_boxes, gt_bx_h),
                box_iou(obj_boxes, gt_bx_o)
                ) >= self.fg_iou_thresh).unbind(1)
            
            if len(x) != 0:
                x = torch.as_tensor(list(set(x.cpu().numpy()))).to(x.device)
            if len(x) == 0: 
                # print(x,y)
                # self.count += 1
                pass
            
            no_pair_x =  list(set(np.arange(len(sub_boxes)).tolist()) - set(x.cpu().numpy().tolist()))
            # x_keep = x_keep[x]
            # y_keep = y_keep[x]
            
            lens = local_features.shape[0]
            feat1 = local_features[:lens//3,:] # human features
            feat2 = local_features[lens//3:lens//3*2,:] # object features
            feat3 = local_features[lens//3*2:,:]  # union features
            obj_union_cat = torch.cat([feat2,feat3],dim=1)
            hum_obj_union_cat = torch.cat([feat1, feat2,feat3],dim=1)
            obj_union_cat = obj_union_cat / obj_union_cat.norm(dim=-1, keepdim=True)
            hum_obj_union_cat = hum_obj_union_cat / hum_obj_union_cat.norm(dim=-1, keepdim=True)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            feat3 = feat3 / feat3.norm(dim=-1, keepdim=True)


            score_h = feat1 @ self.human_clusters.t()
            score_h /= score_h.norm(dim=-1, keepdim=True)
            score_o = feat2 @ self.object_clusters.t()
            score_o /= score_o.norm(dim=-1, keepdim=True)
            # pdb.set_trace()
            score = torch.cat((score_o, score_h), dim=1) ## dim: C1 + C2

            f_cluster = score

            feat_human_obj_union = torch.cat((feat1, feat2,feat3), dim=-1)
            feat_obj_union = torch.cat((feat2,feat3), dim=-1)
            feat_hum_union = torch.cat((feat1,feat3), dim=-1)

            if self.feature == 'hum_uni':
                f_vis = feat_hum_union
            elif self.feature == 'obj_uni':
                f_vis = feat_obj_union
            elif self.feature == 'hum_obj_uni':
                f_vis = feat_human_obj_union
            elif self.feature == 'union':
                f_vis = feat3
            elif self.feature == 'hum_obj':
                f_vis = torch.cat((feat1, feat2), dim=-1)
            else:
                print(f"[ERROR]: feature_type {self.feature} not implemented yet")

            if self._global_norm:
                f_vis /= f_vis.norm(dim=-1, keepdim=True)
            # pdb.set_trace()
            if self.branch == 'F_cluster':
                logits = (f_cluster @ self.hoi_clusters.t()) / 2
                v_min, _ = torch.min(logits, dim=1, keepdim=True)
                v_max, _ = torch.max(logits, dim=1, keepdim=True)
                logits = (logits - v_min) / (v_max - v_min)
                # logits = (100*(logits - 1)).exp()
            elif self.branch == 'F_vis':
                # pdb.set_trace()
                logits = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
            elif self.branch == 'F_cluster_semantic_label':
                logits = f_cluster @ self.hoi_clusters.t()
                v, indices = logits.topk(k=self.topk, dim=-1)
                indices = indices.to(logits.device)
                semantic_label = torch.stack([ nn.functional.softmax(logits[i][indices[i]] / self.temperature) @ self.text_embedding[indices[i]] for i in range(logits.shape[0])], dim=0)
                semantic_label /= semantic_label.norm(dim=-1, keepdim=True)
                logits = semantic_label @ self.text_embedding.t()
            elif self.branch == 'text_only':
                logits = feat3 @ self.text_embedding.t()
            elif self.branch == 'vis+text':
                # pdb.set_trace()
                logits_v = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                logits_v /= 2

                logits_t = feat3 @ self.text_embedding.t()

                logits = self.alpha * logits_v + logits_t
            elif self.branch == 'verb':
                # logits_verb = ((f_vis @ self.verb_cache_models.t()) @ self.verb_one_hots)  / self.verb_sample_lens
                # logits_verb = logits_verb[:, self.HOI_IDX_TO_ACT_IDX] 
                # # pdb.set_trace()
                # logits = logits_verb
                pass

            elif self.branch == 'vis+text+verb':
                pass
                # logits_v = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                # logits_v /= 3

                # logits_t = feat3 @ self.text_embedding.t()
                
                # new_logits = ((f_vis @ self.verb_cache_models.t()) @ self.verb_one_hots)  / self.verb_sample_lens
                # new_logits /= 3
                # logits_verb = new_logits[:, self.HOI_IDX_TO_ACT_IDX] 
                # # pdb.set_trace()
                # logits = logits_v + logits_t + self.alpha * logits_verb
            
            elif self.branch == 'FLL': ## LY 原版 只有Lp
                # pdb.set_trace()
                logits_v = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                logits_v /= 3

                logits_Lp = ((f_vis @ self.cache_models.t()) @ self.Lp_one_hots) /self.Lp_sample_lens
                logits_Lp = logits_Lp[:, self.HOI_IDX_TO_ACT_IDX] 
                logits_Lp /= 2

                logits = 0.5* logits_Lp + logits_v
            elif self.branch == 'FLL+text': ## LY 原版 只有Lp
                # pdb.set_trace()
                logits_v = ((f_vis @ self.cache_models.t()) @ self.one_hots) /self.sample_lens 
                logits_v /= 3

                logits_Lp = ((f_vis @ self.cache_models.t()) @ self.Lp_one_hots) /self.Lp_sample_lens
                logits_Lp = logits_Lp[:, self.HOI_IDX_TO_ACT_IDX] 
                logits_Lp /= 2

                logits_t = feat3 @ self.text_embedding.t()

                logits = 0.5*logits_Lp + logits_t + logits_v
                
            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            if self.post_process:
                '''
                To do
                '''
                # pdb.set_trace()
                obj_l = labels[y_keep].unsqueeze(1).repeat(1,600)
                hoi_l = torch.as_tensor(self.HOI_IDX_TO_OBJ_IDX).unsqueeze(0).repeat(logits.shape[0],1).to(logits.device)
                mask = obj_l == hoi_l
                logits_ = logits.masked_fill(mask == False, float('-inf'))
                logits = (logits_/0.1).softmax(-1)

                logits_no_interaction = (logits_/0.1).softmax(-1)

                indexes_x = torch.arange(logits.shape[0])
                indexes_y = self.obj_to_no_interaction[labels[y_keep]]
                
                no_interaction_logits = (1- logits_no_interaction[indexes_x,indexes_y]).unsqueeze(1)

                # no_pair_logits = logits[no_pair_x] * prior_collated[0].prod(0)[no_pair_x]
                no_pair_logits = logits[no_pair_x] 
                top1_no_pair = no_pair_logits.topk(1)[1]
                for no_i in top1_no_pair:
                    self.count_nopair+=1
                    if no_i in self.no_interaction_indexes:
                        self.r1_nopair +=1
                # pair_logits = logits[x] * prior_collated[0].prod(0)[x]
                pair_logits = logits[x]
                top1_pair = pair_logits.topk(1)[1]
                for i in top1_pair:
                    self.count_pair += 1
                    if i in self.no_interaction_indexes:
                        self.r1_pair +=1

                logits = logits * no_interaction_logits

            # all_logits.append(logits[x])
            all_logits.append(logits)
            # pdb.set_trace()
        return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated
        
    

    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training 
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        # pdb.set_trace()
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
               
        # labels[x, targets['labels'][y]] = 1
        labels[x, targets['hoi'][y]] = 1
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets): ### loss
        
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        # pdb.set_trace()
        logits = torch.cat(logits)
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        # pdb.set_trace()
        
        n_p = len(torch.nonzero(labels))
    
        # print(n_p)
        # pdb.set_trace()
        # if n_p == 0:
        #     pdb.set_trace()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
            )
        # pdb.set_trace()
        # print(loss)
        # if loss.isnan():
        #     pdb.set_trace()
        # loss = binary_focal_loss_with_logits(
        #     logits, labels, reduction='sum',
        #     alpha=self.alpha, gamma=self.gamma
        # )
        # print(loss)
        return loss / n_p

    def prepare_region_proposals(self, results, hidden_states): ## √ detr extracts the human-object pairs
        region_props = []
        for res, hs in zip(results, hidden_states):
            sc, lb, bx = res.values()

            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            hs = hs[keep].view(-1, 256)

            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep]
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            # scores = torch.sigmoid(lg[x, y])
            scores = lg[x, y]

            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores * pr[x, y], labels=y,
                objects=obj[x], size=size
            ))

        return detections
    
    def get_region_proposals(self, results): ##  √√√
        region_props = []
        for res in results:
            # pdb.set_trace()
            bx = res['ex_bbox']
            sc = res['ex_scores']
            lb = res['ex_labels']
            hs = res['ex_hidden_states']
            ms = res['ex_mask']
            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human.sum(); n_object = len(lb) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                # keep_h = keep[keep_h]
                keep_h = hum

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                # keep_o = keep[keep_o]
                keep_o = obj

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
                hidden_states=hs[keep],
                mask = ms[keep]
            ))

        return region_props
        
    def get_targets_pairs(self, targets): ### xxxxxxxxxx
        region_targets = {}
        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        for tar in targets:
            # pdb.set_trace()
            gt_bx_h = self.recover_boxes(tar['boxes_h'], tar['size'])
            gt_bx_o = self.recover_boxes(tar['boxes_o'], tar['size'])
            verbs = tar['labels']
            filename = tar['filename']
            region_targets['filename'] = dict(
                boxes_h=gt_bx_h,
                boxes_o=gt_bx_o,
                verbs=verbs,
            )

        return region_targets
    def get_prior(self, region_props,image_size): ##  for adapter module training
        
        max_feat = 512 + 5
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_w, img_h = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            unary_tokens = props['hidden_states']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                # pdb.set_trace()
                print(n_h,n)
                # sys.exit()
            # pdb.set_trace()
            object_embs = self.object_embedding[labels]
            priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            priors[b_idx,:n,5:] = object_embs
            mask[b_idx,:n] = False
        # pdb.set_trace()
        priors = self.priors(priors)
        return (priors, mask)

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self._load_features:
            image_sizes = torch.stack([t['size'] for t in targets])
            region_props = self.get_region_proposals(targets)
            cls_feature = images[0]
            logits, prior, bh, bo, objects = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
            boxes = [r['boxes'] for r in region_props]   
        else:
            if self.training and targets is None:
                raise ValueError("In training mode, targets should be passed")
            image_sizes = torch.as_tensor([
                im.size()[-2:] for im in images
            ], device=images[0].device)
            region_props = self.get_region_proposals(targets)  # exhaustively generate the human-object pairs from the detr results
            feat_local_old = self.clip_model.encode_image(images[0])
            feat_local = feat_local_old[:,1:,:].transpose(1,2).view(feat_local_old.shape[0],-1, 7, 7).float()
            cls_feature = feat_local_old[:,0,:]
            # use the gt crop
            if self.evaluate_type == 'gt': 
                if self.use_type == 'crop':
                    logits, prior, bh, bo, objects, boxes = self.compute_roi_embeddings_targets(cls_feature.unsqueeze(0), image_sizes, targets)
                else: #### ignore 
                    logits, prior, bh, bo, objects = self.compute_roi_embeddings(feat_local, image_sizes, region_props)
            elif self.evaluate_type == 'detr':      
                logits, prior, bh, bo, objects = self.compute_crop_embeddings(cls_feature.unsqueeze(0), image_sizes, region_props, targets)
                boxes = [r['boxes'] for r in region_props]   
            
            if self.training:
                interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets)
                loss_dict = dict(
                    interaction_loss=interaction_loss
                )
                return loss_dict

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes)
        return detections

def build_detector(args, class_corr):
    detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    predictor = torch.nn.Linear(args.repr_dim * 2, args.num_classes)
    interaction_head = InteractionHead(
        predictor, args.hidden_dim, args.repr_dim,
        detr.backbone[0].num_channels,
        args.num_classes, args.human_idx, class_corr
    )
    
    if args.visual_mode == 'res':
        clip_head = CLIP_ResNet(embed_dim=args.clip_visual_output_dim,
                                image_resolution=args.clip_visual_input_resolution,
                                vision_layers=args.clip_visual_layers,
                                vision_width=args.clip_visual_width,
                                vision_patch_size=args.clip_visual_patch_size,
                                context_length=args.clip_text_context_length,
                                transformer_width=args.clip_text_transformer_width,
                                transformer_heads=args.clip_text_transformer_heads,
                                transformer_layers=args.clip_text_transformer_layers)
    elif args.visual_mode == 'vit':
        clip_head = CLIP_ResNet(embed_dim=args.clip_visual_output_dim_vit,
                                image_resolution=args.clip_visual_input_resolution_vit,
                                vision_layers=args.clip_visual_layers_vit,
                                vision_width=args.clip_visual_width_vit,
                                vision_patch_size=args.clip_visual_patch_size_vit,
                                context_length=args.clip_text_context_length_vit,
                                transformer_width=args.clip_text_transformer_width_vit,
                                transformer_heads=args.clip_text_transformer_heads_vit,
                                transformer_layers=args.clip_text_transformer_layers_vit)
    detector = UPT(
        detr, postprocessors['bbox'], clip_head, args.clip_dir_vit,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        topk = args.topk,
        branch = args.branch,
        post_process = args.post_process,
        use_outliers = args.use_outliers,
        use_less_confident = args.use_less_confident,
        num_shot = args.num_shot,
    )
    return detector
