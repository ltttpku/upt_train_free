"""
Utilities

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from code import interact
from fileinput import filename
from locale import normalize
import os
import torch
import pickle
import numpy as np
import scipy.io as sio
import json

from torchvision.transforms import Resize, CenterCrop

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

from vcoco.vcoco import VCOCO
from hicodet.hicodet import HICODet
import sys
sys.path.append('../pocket/pocket')
import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, BoxPairAssociation

import sys
sys.path.append('detr')
import datasets.transforms_clip as T
import pdb
import copy 
import pickle
import torch.nn.functional as F
import clip
from util import box_ops
from PIL import Image

def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
        
        # images_clip.append(im_clip)
    return images, targets

class DataFactory(Dataset):
    def __init__(self, name, partition, data_root,few_shot_pickle):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)
        
        if name == 'hicodet':
            self._load_features= True
            self._text_features = pickle.load(open('inference_features_vit16.p','rb'))
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            if partition == 'train2015':
                self.anno_bbox = pickle.load(open('hico_train_bbox_max50.p','rb'))
            else:
                self.anno_bbox = pickle.load(open('hico_test_bbox.p','rb'))
            # pdb.set_trace()
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )

        # add clip normalization 
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        normalize_clip = T.Compose([
            T.ToTensor(),
            T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        normalize_clip_1 = T.ToTensor()
        normalize_clip_2 = T.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        if partition.startswith('train'):
            self.transforms = [T.Compose([
                T.RandomHorizontalFlip(),
                T.ColorJitter(.4, .4, .4),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=1333),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=1333),
                    ]))
                    ]),
        normalize, normalize_clip,
        T.Compose([
                 T.IResize([224,224])
                # T.IResize(224),
                # T.CenterCrop([224,224])
            ])
        ]
        else:   
            self.transforms = [T.Compose([
                T.RandomResize([800], max_size=1333),
            ]),
            normalize, normalize_clip,
            T.Compose([
                 T.IResize([224,224])
                # T.IResize(224),
                # T.CenterCrop([224,224])
            ]),
            normalize_clip_1,
            normalize_clip_2

            ]

        self.partition = partition
        self.name = name
        self.count=0

        device = "cuda"
        _, self.process = clip.load('ViT-B/16', device=device)

        zero_shot_rare_first = [509, 279, 280, 402, 504, 286, 499, 498, 289, 485, 303, 311, 325, 439, 351, 358, 66, 427, 379, 418,
                   70, 416,
                   389, 90, 395, 76, 397, 84, 135, 262, 401, 592, 560, 586, 548, 593, 526, 181, 257, 539, 535, 260, 596,
                   345, 189,
                   205, 206, 429, 179, 350, 405, 522, 449, 261, 255, 546, 547, 44, 22, 334, 599, 239, 315, 317, 229,
                   158, 195,
                   238, 364, 222, 281, 149, 399, 83, 127, 254, 398, 403, 555, 552, 520, 531, 440, 436, 482, 274, 8, 188,
                   216, 597,
                   77, 407, 556, 469, 474, 107, 390, 410, 27, 381, 463, 99, 184, 100, 292, 517, 80, 333, 62, 354, 104,
                   55, 50,
                   198, 168, 391, 192, 595, 136, 581]
        zero_shot_non_rare_first = [38, 41, 20, 18, 245, 11, 19, 154, 459, 42, 155, 139, 60, 461, 577, 153, 582, 89, 141, 576, 75,
                       212, 472, 61,
                       457, 146, 208, 94, 471, 131, 248, 544, 515, 566, 370, 481, 226, 250, 470, 323, 169, 480, 479,
                       230, 385, 73,
                       159, 190, 377, 176, 249, 371, 284, 48, 583, 53, 162, 140, 185, 106, 294, 56, 320, 152, 374, 338,
                       29, 594, 346,
                       456, 589, 45, 23, 67, 478, 223, 493, 228, 240, 215, 91, 115, 337, 559, 7, 218, 518, 297, 191,
                       266, 304, 6, 572,
                       529, 312, 9, 308, 417, 197, 193, 163, 455, 25, 54, 575, 446, 387, 483, 534, 340, 508, 110, 329,
                       246, 173, 506,
                       383, 93, 516, 64]
        self.zero_shot = True
        zero_shot_type = 'rare_first'
        if zero_shot_type=='rare_first':
            zero_indexes = zero_shot_rare_first
        else:
            zero_indexes = zero_shot_non_rare_first
        count = 0
        if partition.startswith('train'):
            self.train = True
            # pdb.set_trace()
            
            # self.few_shot_anno = pickle.load(open('/work/yangl/hoi/leiting/upt_train_free/few_shot_pickle/few_shot_4_2189.p','rb'))
            self.few_shot_anno = pickle.load(open(load_few_shot_pickle,'rb'))
            self.sampled_idx = [anno['original_idx'] for anno in self.few_shot_anno]
            new_idx = []
            if self.zero_shot:
                
                for k, anno in enumerate(self.few_shot_anno):
                    hois = anno['hoi']
                    save_indexes = []
                    for i, hoi in enumerate(hois):
                        if hoi not in zero_indexes: save_indexes.append(i)
                        else: count +=1 
                    # pdb.set_trace()
                    anno['boxes_h'] = (np.array(anno['boxes_h'])[save_indexes]).tolist()
                    anno['boxes_o'] = (np.array(anno['boxes_o'])[save_indexes]).tolist()
                    anno['hoi'] =  (np.array(anno['hoi'])[save_indexes]).tolist()
                    anno['object'] =  (np.array(anno['object'])[save_indexes]).tolist()
                    anno['verb'] =  (np.array(anno['verb'])[save_indexes]).tolist()
                    anno['pair_with_original'] =  (np.array(anno['pair_with_original'])[save_indexes]).tolist()
                    if len(anno['boxes_h']) != 0:
                        new_idx.append(k)
                    else:
                        pass
                new_few_shot_anno = []
                new_sampled_idx = []
                for ind in new_idx:
                    new_few_shot_anno.append(self.few_shot_anno[ind])
                    new_sampled_idx.append(self.sampled_idx[ind])
                self.few_shot_anno = new_few_shot_anno
                self.sampled_idx = new_sampled_idx
            # pdb.set_trace()
            print("all training samples: {}, all sampled indexes: {}".format(len(self.few_shot_anno),len(self.sampled_idx)))

        else:
            self.train = False

    def __len__(self):
        if self.train:
            return len(self.sampled_idx)
        else:
            return len(self.dataset)
    #@@@## use roi 
    def __getitem__(self, i):
        # pdb.set_trace()
        # (image, target), filename = self.dataset[i]
        # w,h = image.size
        # target['orig_size'] = torch.tensor([h,w])
        
        # anno_bbox_list = self.anno_bbox[filename][0]
        # ex_bbox = anno_bbox_list['boxes']
        # target['ex_bbox'] = torch.as_tensor(ex_bbox)
        # target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
        # target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
        # target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
        
        if self.train:
            # pdb.set_trace()
            
            idx_ori = self.sampled_idx[i]
            refine_anno = self.few_shot_anno[i]
            pair_with_real = refine_anno['pair_with_original']

            # pdb.set_trace()
            index = self.dataset._idx.index(idx_ori)

            
            (image, target), filename = self.dataset[i]
            target['verb'] = target['verb'][pair_with_real]
            target['boxes_h'] = target['boxes_h'][pair_with_real]
            target['boxes_o'] = target['boxes_o'][pair_with_real]
            target['object'] = target['object'][pair_with_real]
            target['hoi'] = target['hoi'][pair_with_real]

        else:
            (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])

        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
        # test
        
        image_0, target_0 = self.transforms[0](image, target)
        image_org, _ = self.transforms[1](image_0, target_0)
        pdb.set_trace()

        image_0, target_0 = self.transforms[3](image, target_0)
        image_clip, target = self.transforms[2](image_0, target_0)
        if image_0.size[-1] >224 or image_0.size[-2] >224:print(image_0.size)
        target['filename'] = filename

        # mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
        # for i in range(len(target['ex_bbox'])):
        #     t = target['ex_bbox'][i].clamp(0,224).int()
        #     mask[i, t[1]:t[3], t[0]:t[2]] = 1
        # # pdb.set_trace()
        # assert mask.shape[0] != 0
        # mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
        # target['ex_mask'] = mask

        return (image_org,image_clip), target


    ####  use gt box and class type: crop 
    # def f(self, i):
    #     # pdb.set_trace()
    #     (image, target), filename = self.dataset[i]
    #     w,h = image.size
    #     target['orig_size'] = torch.tensor([h,w])
        
    #     anno_bbox_list = self.anno_bbox[filename][0]
    #     ex_bbox = anno_bbox_list['boxes']
    #     target['ex_bbox'] = torch.as_tensor(ex_bbox)
    #     target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
    #     target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
    #     target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
        
    #     if self.name == 'hicodet':
    #         target['labels'] = target['verb']
    #         # Convert ground truth boxes to zero-based index and the
    #         # representation from pixel indices to coordinates
    #         target['boxes_h'][:, :2] -= 1
    #         target['boxes_o'][:, :2] -= 1
    #     else:
    #         target['labels'] = target['actions']
    #         target['object'] = target.pop('objects')
    #     # test

    #     lt = torch.min(target['boxes_h'][...,:2],target['boxes_o'][...,:2])
    #     rb = torch.max(target['boxes_h'][..., 2:], target['boxes_o'][..., 2:])
    #     crop_size=torch.cat([lt,rb],dim=-1).numpy()
    #     crop_size_object = target['boxes_o'].numpy()
    #     crop_size_human = target['boxes_h'].numpy()
    #     all_images = []
    #     all_objects = []
    #     all_human = []
    #     for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
    #         new_img = image.crop(crop_s)
    #         all_images.append(self.process(new_img))
    #         new_img = image.crop(crop_s_o)
    #         all_objects.append(self.process(new_img))
    #         new_img = image.crop(crop_s_h)
    #         all_human.append(self.process(new_img))
        
    #     all_images = torch.stack(all_images)
    #     all_images_object = torch.stack(all_objects)
    #     all_images_human = torch.stack(all_human)

    #     all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)

    #     image_0, target_0 = self.transforms[3](image, target)
    #     image_clip, target = self.transforms[2](image_0, target_0)
    #     if image_0.size[-1] >224 or image_0.size[-2] >224:print(image_0.size)
    #     target['filename'] = filename

    #     ### ignore 
    #     mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
    #     for i in range(len(target['ex_bbox'])):
    #         t = target['ex_bbox'][i].clamp(0,224).int()
    #         mask[i, t[1]:t[3], t[0]:t[2]] = 1
    #     # pdb.set_trace()
    #     assert mask.shape[0] != 0
    #     mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
    #     target['ex_mask'] = mask

    #     return all_images,target

    ### detr extract bbox 
    '''
    def __getitem__(self, i):
        # pdb.set_trace()
        (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])
        
        anno_bbox_list = self.anno_bbox[filename][0]
        ex_bbox = anno_bbox_list['boxes']
        target['ex_bbox'] = torch.as_tensor(ex_bbox)
        target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
        target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
        target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
        # pdb.set_trace()
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')
        # test
        crop_size_human, crop_size_object, crop_size = self.get_region_proposals(target,image_h=image.size[1], image_w=image.size[0])
        # pdb.set_trace()
        crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
        all_images = []
        all_objects = []
        all_human = []
        
        for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
            new_img = image.crop(crop_s)
            all_images.append(self.process(new_img))
            new_img = image.crop(crop_s_o)
            all_objects.append(self.process(new_img))
            new_img = image.crop(crop_s_h)
            all_human.append(self.process(new_img))
        
        all_images = torch.stack(all_images)
        all_images_object = torch.stack(all_objects)
        all_images_human = torch.stack(all_human)
        # all_images = torch.cat([all_images_object,all_images],dim=0)
        all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
        # pdb.set_trace()
        image_0, target_0 = self.transforms[3](image, target)
        image_clip, target = self.transforms[2](image_0, target_0)
        if image_0.size[-1] >224 or image_0.size[-2] >224:print(image_0.size)
        target['filename'] = filename

        mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
        for i in range(len(target['ex_bbox'])):
            t = target['ex_bbox'][i].clamp(0,224).int()
            mask[i, t[1]:t[3], t[0]:t[2]] = 1
        # pdb.set_trace()
        assert mask.shape[0] != 0
        mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
        target['ex_mask'] = mask

        return all_images, target
    '''
    ##  padding zeros
    def __getitem__(self, i):
        # pdb.set_trace()
        (image, target), filename = self.dataset[i]
        w,h = image.size
        target['orig_size'] = torch.tensor([h,w])
        
        anno_bbox_list = self.anno_bbox[filename][0]
        ex_bbox = anno_bbox_list['boxes']
        target['ex_bbox'] = torch.as_tensor(ex_bbox)
        target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
        target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
        target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
        # pdb.set_trace()
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        if self._load_features:
            all_images = torch.as_tensor(self._text_features[filename])
        else:
            # test
            crop_size_human, crop_size_object, crop_size = self.get_region_proposals(target,image_h=image.size[1], image_w=image.size[0])
            # pdb.set_trace()
            crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
            all_images = []
            all_objects = []
            all_human = []
            
            for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
                new_img = image.crop(crop_s)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_images.append(self.process(new_img))
                new_img = image.crop(crop_s_o)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_objects.append(self.process(new_img))
                new_img = image.crop(crop_s_h)
                new_img = self.expand2square(new_img,(0,0,0)) #
                all_human.append(self.process(new_img))
            
            all_images = torch.stack(all_images)
            all_images_object = torch.stack(all_objects)
            all_images_human = torch.stack(all_human)
            # all_images = torch.cat([all_images_object,all_images],dim=0)
            all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
        # pdb.set_trace()
        image_0, target_0 = self.transforms[3](image, target)
        image_clip, target = self.transforms[2](image_0, target_0)
        if image_0.size[-1] >224 or image_0.size[-2] >224:print(image_0.size)
        target['filename'] = filename

        mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
        for i in range(len(target['ex_bbox'])):
            t = target['ex_bbox'][i].clamp(0,224).int()
            mask[i, t[1]:t[3], t[0]:t[2]] = 1
        # pdb.set_trace()
        assert mask.shape[0] != 0
        mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
        target['ex_mask'] = mask

        return all_images, target
    ###  ignore 
    # def __getitem__(self, i):
    #     # pdb.set_trace()
    #     (image, target), filename = self.dataset[i]
    #     w,h = image.size
    #     target['orig_size'] = torch.tensor([h,w])
        
    #     anno_bbox_list = self.anno_bbox[filename][0]
    #     ex_bbox = anno_bbox_list['boxes']
    #     target['ex_bbox'] = torch.as_tensor(ex_bbox)
    #     target['ex_scores'] = torch.as_tensor(anno_bbox_list['scores'])
    #     target['ex_labels'] = torch.as_tensor(anno_bbox_list['labels'])
    #     target['ex_hidden_states'] = torch.as_tensor(anno_bbox_list['hidden_states'])
    #     if self.name == 'hicodet':
    #         target['labels'] = target['verb']
    #         # Convert ground truth boxes to zero-based index and the
    #         # representation from pixel indices to coordinates
    #         target['boxes_h'][:, :2] -= 1
    #         target['boxes_o'][:, :2] -= 1
    #     else:
    #         target['labels'] = target['actions']
    #         target['object'] = target.pop('objects')
    #     # test
    #     crop_size_human, crop_size_object, crop_size = self.get_region_proposals(target,image_h=image.size[1], image_w=image.size[0])
    #     # pdb.set_trace()
    #     # crop_size_human, crop_size_object, crop_size = crop_size_human.numpy(), crop_size_object.numpy(), crop_size.numpy()
    #     all_images = []
    #     all_objects = []
    #     all_human = []
    #     # for crop_s, crop_s_o, crop_s_h in zip(crop_size,crop_size_object,crop_size_human):
    #     #     new_img = image.crop(crop_s)
    #     #     all_images.append(self.process(new_img))
    #     #     # new_img = self.transforms[-1](new_img,None)[0]
    #     #     # new_img_tensor = self.transforms[2](new_img, None)[0]
    #     #     # all_images.append(new_img_tensor)
    #     #     # new_img = image.crop(crop_s_o)
    #     #     # all_objects.append(self.process(new_img))
    #     #     # new_img = image.crop(crop_s_h)
    #     #     # all_human.append(self.process(new_img))
    #     # # pdb.set_trace()
    #     # all_images = torch.stack(all_images)
    #     # # all_images_object = torch.stack(all_objects)
    #     # # all_images_human = torch.stack(all_human)
    #     # # all_images = torch.cat([all_images_object,all_images],dim=0)
    #     # # all_images = torch.cat([all_images_human,all_images_object,all_images],dim=0)
        
    #     w,h =image.size
    #     masks = torch.zeros((len(crop_size),h,w),dtype=torch.float)
    #     mask_fill = [c for c in (0.48145466, 0.4578275, 0.40821073)]
    #     # mask_fill = [0,0,0]
    #     # x1, y1, x2, y2 = crop_size.unbind(-1)
    #     # x1 = torch.clamp(x1,0,w)
    #     # y1 = torch.clamp(y1,0,h)
    #     # x2 = torch.clamp(x2,0,w)
    #     # y2 = torch.clamp(y2,0,h)
    #     image_tensor = self.transforms[-2](image, None)[0]

    #     human_center = box_ops.box_xyxy_to_cxcywh(crop_size_human).int()
    #     object_center = box_ops.box_xyxy_to_cxcywh(crop_size_object).int()

    #     x_min = torch.min(human_center[..., 0], object_center[..., 0]) # left point
    #     x_max = torch.max(human_center[..., 0], object_center[..., 0]) # right point
    #     y_min = torch.min(human_center[..., 1], object_center[..., 1]) # left point
    #     y_max = torch.max(human_center[..., 1], object_center[..., 1]) # right point


    #     new_image_all = []
        
    #     # new_image = torch.cat([image_tensor.new_full((1, y2 - y1, x2 - x1), fill_value=val) for val in mask_fill])
    #     indexes = torch.arange(len(crop_size))
    #     for ind, crop_s, crop_s_o, crop_s_h in zip(indexes, crop_size,crop_size_object,crop_size_human):
    #         # new_image = self.transforms[4](image, None)[0]
    #         new_image = torch.cat([image_tensor.new_full((1,image_tensor.shape[1], image_tensor.shape[2]), fill_value=val) for val in mask_fill])
    #         # new_image_all.append(new_image)
    #         masks[ind][crop_s_h[1]:crop_s_h[3],crop_s_h[0]:crop_s_h[2]] = 1
    #         masks[ind][crop_s_o[1]:crop_s_o[3],crop_s_o[0]:crop_s_o[2]] = 1
    #         masks[ind][y_min[ind]:y_max[ind],x_min[ind]:x_max[ind]] = 1
    #         retain_image = ((1-masks[ind]).unsqueeze(0) * new_image)[:,crop_s[1]:crop_s[3],crop_s[0]:crop_s[2]]
    #         save_image = (image_tensor * masks[ind])[:,crop_s[1]:crop_s[3],crop_s[0]:crop_s[2]]
    #         normalize_image = self.transforms[-1](save_image+retain_image, None)[0]
    #         all_images.append(normalize_image)
    #     # pdb.set_trace()
    #     all_images = [F.interpolate(r[None,...], size=(224, 224), mode="bicubic") for r in all_images]
    #     all_images = torch.cat(all_images)
        
        


    #     image_0, target_0 = self.transforms[3](image, target)
    #     image_clip, target = self.transforms[2](image_0, target_0)
    #     if image_0.size[-1] >224 or image_0.size[-2] >224:print(image_0.size)
    #     target['filename'] = filename

    #     mask = torch.zeros((len(target['ex_bbox']), 224, 224), dtype=torch.bool)
    #     for i in range(len(target['ex_bbox'])):
    #         t = target['ex_bbox'][i].clamp(0,224).int()
    #         mask[i, t[1]:t[3], t[0]:t[2]] = 1
    #     # pdb.set_trace()
    #     assert mask.shape[0] != 0
    #     mask = F.interpolate(mask[None].float(), size=(7,7)).to(torch.bool)[0]
    #     target['ex_mask'] = mask

    #     return all_images, target

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
            
    def get_region_proposals(self, results,image_h, image_w):
        human_idx = 0
        min_instances = 3
        max_instances = 15
        region_props = []
        # for res in results:
        # pdb.set_trace()
        bx = results['ex_bbox']
        sc = results['ex_scores']
        lb = results['ex_labels']
        hs = results['ex_hidden_states']
        is_human = lb == human_idx
        hum = torch.nonzero(is_human).squeeze(1)
        obj = torch.nonzero(is_human == 0).squeeze(1)
        n_human = is_human.sum(); n_object = len(lb) - n_human
        # Keep the number of human and object instances in a specified interval
        device = torch.device('cpu')
        if n_human < min_instances:
            keep_h = sc[hum].argsort(descending=True)[:min_instances]
            keep_h = hum[keep_h]
        elif n_human > max_instances:
            keep_h = sc[hum].argsort(descending=True)[:max_instances]
            keep_h = hum[keep_h]
        else:
            # keep_h = torch.nonzero(is_human[keep]).squeeze(1)
            # keep_h = keep[keep_h]
            keep_h = hum

        if n_object < min_instances:
            keep_o = sc[obj].argsort(descending=True)[:min_instances]
            keep_o = obj[keep_o]
        elif n_object > max_instances:
            keep_o = sc[obj].argsort(descending=True)[:max_instances]
            keep_o = obj[keep_o]
        else:
            # keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
            # keep_o = keep[keep_o]
            keep_o = obj

        keep = torch.cat([keep_h, keep_o])

        boxes=bx[keep]
        scores=sc[keep]
        labels=lb[keep]
        hidden_states=hs[keep]
        is_human = labels == human_idx
            
        n_h = torch.sum(is_human); n = len(boxes)
        # Permute human instances to the top
        if not torch.all(labels[:n_h]==human_idx):
            h_idx = torch.nonzero(is_human).squeeze(1)
            o_idx = torch.nonzero(is_human == 0).squeeze(1)
            perm = torch.cat([h_idx, o_idx])
            boxes = boxes[perm]; scores = scores[perm]
            labels = labels[perm]; unary_tokens = unary_tokens[perm]
        # Skip image when there are no valid human-object pairs
        if n_h == 0 or n <= 1:
            print(n_h, n)
            # boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
            # prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
            # continue

        # Get the pairwise indices
        x, y = torch.meshgrid(
            torch.arange(n, device=device),
            torch.arange(n, device=device)
        )
        # pdb.set_trace()
        # Valid human-object pairs
        x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
        sub_boxes = boxes[x_keep]
        obj_boxes = boxes[y_keep]
        lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
        rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
        union_boxes = torch.cat([lt,rb],dim=-1)
        sub_boxes[:,0].clamp_(0, image_w)
        sub_boxes[:,1].clamp_(0, image_h)
        sub_boxes[:,2].clamp_(0, image_w)
        sub_boxes[:,3].clamp_(0, image_h)

        obj_boxes[:,0].clamp_(0, image_w)
        obj_boxes[:,1].clamp_(0, image_h)
        obj_boxes[:,2].clamp_(0, image_w)
        obj_boxes[:,3].clamp_(0, image_h)

        union_boxes[:,0].clamp_(0, image_w)
        union_boxes[:,1].clamp_(0, image_h)
        union_boxes[:,2].clamp_(0, image_w)
        union_boxes[:,3].clamp_(0, image_h)
      
        # region_props.append(dict(
        #     boxes=bx[keep],
        #     scores=sc[keep],
        #     labels=lb[keep],
        #     hidden_states=hs[keep],
        #     mask = ms[keep]
        # ))

        # return sub_boxes.int(), obj_boxes.int(), union_boxes.int()
        return sub_boxes, obj_boxes, union_boxes

    def get_union_mask(self, bbox, image_size):
        n = len(bbox)
        masks = torch.zeros
        pass
class CacheTemplate(defaultdict):
    """A template for VCOCO cached results """
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            self[k] = v
    def __missing__(self, k):
        seg = k.split('_')
        # Assign zero score to missing actions
        if seg[-1] == 'agent':
            return 0.
        # Assign zero score and a tiny box to missing <action,role> pairs
        else:
            return [0., 0., .1, .1, 0.]

class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, dataloader, max_norm=0, num_classes=117, **kwargs):
        super().__init__(net, None, dataloader, **kwargs)
        self.max_norm = max_norm
        self.num_classes = num_classes
        self.test_loader = kwargs['test_loader']
        self.anno_interaction = kwargs['anno_interaction']
        # self.cache_dir = kwargs['cache_dir']
    def _on_each_iteration(self):
        loss_dict = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        if loss_dict['interaction_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.optimizer.zero_grad(set_to_none=True)
        self._state.loss.backward()
        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._state.net.parameters(), self.max_norm)
        self._state.optimizer.step()
    @torch.no_grad()
    def _test_hico(self):
        if self._rank == 0:
            ap = self.test_hico(self.test_loader)
            num_anno = torch.as_tensor(self.anno_interaction)
            rare = torch.nonzero(num_anno < 10).squeeze(1)
            non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
            print(
                f"The mAP is {ap.mean():.4f},"
                f" rare: {ap[rare].mean():.4f},"
                f" none-rare: {ap[non_rare].mean():.4f},"
                
            )
            log_stats= f"The mAP is {ap.mean():.4f}, rare: {ap[rare].mean():.4f}, none-rare: {ap[non_rare].mean():.4f},"
        
            # pdb.set_trace()
            with open(self._cache_dir + "/log.txt","a") as f:
                f.write(log_stats + "\n")

    @torch.no_grad()
    def test_hico(self, dataloader):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        associate = BoxPairAssociation(min_iou=0.5)
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        # pdb.set_trace()
        interaction_to_verb = torch.as_tensor(dataset.interaction_to_verb)
        
        meter = DetectionAPMeter(
            600, nproc=1,
            num_gt=dataset.anno_interaction,
            algorithm='11P'
        )

        dicts = {}
        
        for batch in tqdm(dataloader):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            # pdb.set_trace()
            outputs = net(inputs,batch[1])

            
            # continue
            # Skip images without detections
            
            if outputs is None or len(outputs) == 0:
                continue
            # # Batch size is fixed as 1 for inference
            # assert len(output) == 1, f"Batch size is not 1 but {len(outputs)}."
            for output, target in zip(outputs, batch[-1]):
                # output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
                # target = batch[-1][0]
                output = pocket.ops.relocate_to_cpu(output, ignore=True)
                # pdb.set_trace()
                # Format detections
                boxes = output['boxes']
                boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
                objects = output['objects']
                scores = output['scores']
                verbs = output['labels']
                # pdb.set_trace()
                # interactions = verbs
                # verbs = interaction_to_verb[verbs]
                if net.module.class_nums==117:
                    interactions = conversion[objects, verbs]
                else:
                    interactions = verbs
                # Recover target box scale
                gt_bx_h = net.module.recover_boxes(target['boxes_h'], target['size'])
                gt_bx_o = net.module.recover_boxes(target['boxes_o'], target['size'])
                # pdb.set_trace()
                # Associate detected pairs with ground truth pairs
                labels = torch.zeros_like(scores)
                unique_hoi = interactions.unique()
                
                
                for hoi_idx in unique_hoi:
                    gt_idx = torch.nonzero(target['hoi'] == hoi_idx).squeeze(1)
                    det_idx = torch.nonzero(interactions == hoi_idx).squeeze(1)
                    if len(gt_idx):
                        # pdb.set_trace()
                        labels[det_idx] = associate(
                            (gt_bx_h[gt_idx].view(-1, 4),
                            gt_bx_o[gt_idx].view(-1, 4)),
                            (boxes_h[det_idx].view(-1, 4),
                            boxes_o[det_idx].view(-1, 4)),
                            scores[det_idx].view(-1)
                        )
                        # all_det_idxs.append(det_idx)
                
                meter.append(scores, interactions, labels)   # scores human*object*verb, interaction（600), labels
        
        # pdb.set_trace()
        # with open("training_free_files/union_embeddings_cachemodel_crop_invalidpairs.p","wb") as f:
        #     pickle.dump(net.module.dicts,f)
        # with open("union_embeddings_cachemodel_clipcrop.p","wb") as f:
        #     pickle.dump(net.module.dicts,f)
        # with open('hico_train_bbox_max25.p','wb') as f:
        #     pickle.dump(dicts,f)
        # print(count)
        return meter.eval()

    @torch.no_grad()
    def cache_hico(self, dataloader, cache_dir='matlab'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        conversion = torch.from_numpy(np.asarray(
            dataset.object_n_verb_to_interaction, dtype=float
        ))
        object2int = dataset.object_to_interaction

        # Include empty images when counting
        nimages = len(dataset.annotations)
        all_results = np.empty((600, nimages), dtype=object)

        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_idx = dataset._idx[i]
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            objects = output['objects']
            scores = output['scores']
            verbs = output['labels']
            interactions = conversion[objects, verbs]
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            # Convert box representation to pixel indices
            boxes_h[:, 2:] -= 1
            boxes_o[:, 2:] -= 1

            # Group box pairs with the same predicted class
            permutation = interactions.argsort()
            boxes_h = boxes_h[permutation]
            boxes_o = boxes_o[permutation]
            interactions = interactions[permutation]
            scores = scores[permutation]

            # Store results
            unique_class, counts = interactions.unique(return_counts=True)
            n = 0
            for cls_id, cls_num in zip(unique_class, counts):
                all_results[cls_id.long(), image_idx] = torch.cat([
                    boxes_h[n: n + cls_num],
                    boxes_o[n: n + cls_num],
                    scores[n: n + cls_num, None]
                ], dim=1).numpy()
                n += cls_num
        
        # Replace None with size (0,0) arrays
        for i in range(600):
            for j in range(nimages):
                if all_results[i, j] is None:
                    all_results[i, j] = np.zeros((0, 0))
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        # Cache results
        for object_idx in range(80):
            interaction_idx = object2int[object_idx]
            sio.savemat(
                os.path.join(cache_dir, f'detections_{(object_idx + 1):02d}.mat'),
                dict(all_boxes=all_results[interaction_idx])
            )

    @torch.no_grad()
    def cache_vcoco(self, dataloader, cache_dir='vcoco_cache'):
        net = self._state.net
        net.eval()

        dataset = dataloader.dataset.dataset
        all_results = []
        for i, batch in enumerate(tqdm(dataloader)):
            inputs = pocket.ops.relocate_to_cuda(batch[0])
            output = net(inputs)

            # Skip images without detections
            if output is None or len(output) == 0:
                continue
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, f"Batch size is not 1 but {len(output)}."
            output = pocket.ops.relocate_to_cpu(output[0], ignore=True)
            # NOTE Index i is the intra-index amongst images excluding those
            # without ground truth box pairs
            image_id = dataset.image_id(i)
            # Format detections
            boxes = output['boxes']
            boxes_h, boxes_o = boxes[output['pairing']].unbind(0)
            scores = output['scores']
            actions = output['labels']
            # Rescale the boxes to original image size
            ow, oh = dataset.image_size(i)
            h, w = output['size']
            scale_fct = torch.as_tensor([
                ow / w, oh / h, ow / w, oh / h
            ]).unsqueeze(0)
            boxes_h *= scale_fct
            boxes_o *= scale_fct

            for bh, bo, s, a in zip(boxes_h, boxes_o, scores, actions):
                a_name = dataset.actions[a].split()
                result = CacheTemplate(image_id=image_id, person_box=bh.tolist())
                result[a_name[0] + '_agent'] = s.item()
                result['_'.join(a_name)] = bo.tolist() + [s.item()]
                all_results.append(result)

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        with open(os.path.join(cache_dir, 'cache.pkl'), 'wb') as f:
            # Use protocol 2 for compatibility with Python2
            pickle.dump(all_results, f, 2)


if __name__ == '__main__':
    meter = DetectionAPMeter(
            60, #nproc=1,
            # num_gt=dataset.anno_interaction,
            algorithm='11P'
        )
    scores = torch.rand(10000)
    pred = torch.randint(0, 60, (10000,))
    trueorfalse = torch.randint(0, 2, (10000,))
    meter.append(scores, pred, trueorfalse)
    ap = meter.eval()
    mAP = ap.mean()
    print(mAP) ## 0.5537

    meter.reset()
    ## 加上一些 false positive 和 false negative 
    ## (detr bbox和 gt bbox相差大的那部分一定是false positive或者false negative)
    scores = torch.cat([scores, torch.ones(5000) * 0.01], dim=0)
    pred = torch.cat([pred, torch.randint(0, 60, (5000,))], dim=0)
    trueorfalse = torch.cat([trueorfalse, torch.zeros(5000)], dim=0)
    meter.append(scores, pred, trueorfalse)
    ap = meter.eval()
    mAP = ap.mean()
    print(mAP) ## 0.3817

    
