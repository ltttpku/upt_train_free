import json
import random
import pickle
import torch
import argparse
import pdb

parser = argparse.ArgumentParser()

parser.add_argument('--sample_lens', default=2, type=int)
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--save_dir', default='few_shot_pickle/final_sets/', type=str)



args = parser.parse_args()
print(args)

random.seed(args.seed)
anno_file = 'hicodet/instances_train2015.json'
with open(anno_file, 'r') as f:
    anno = json.load(f)

# annotation 
idx = list(range(len(anno['filenames'])))
for empty_idx in anno['empty']:
    idx.remove(empty_idx)
random.shuffle(idx)
# num_anno = [0 for _ in range(self.num_interation_cls)]
# for anno in f['annotation']:
#     for hoi in anno['hoi']:
#         num_anno[hoi] += 1
# pdb.set_trace()
_idx = idx
# _num_anno = num_anno

_anno = anno['annotation']
_filenames = anno['filenames']
_image_sizes = anno['size']
_class_corr = anno['correspondence']
_empty_idx = anno['empty']
_objects = anno['objects']
_verbs = anno['verbs']

# generate few_shot.pickle
sample_lens = args.sample_lens
all_list = []
already_sampled = [0 for i in range(600)]
for k in idx:
    
    anno_one = _anno[k]
    filename = _filenames[k]
    size = _image_sizes[k]
    
    hois = anno_one['hoi']
    objects = anno_one['object']
    verbs = anno_one['verb']
    boxes_h = anno_one['boxes_h']
    boxes_o = anno_one['boxes_o']
    
    
    all_boxes_h = []
    all_boxes_o = []
    all_hois = []
    all_objects = []
    all_verbs = []
    pair_with_original = []
    add_true = False
    for i,hoi in enumerate(hois):
        if already_sampled[hoi] != sample_lens:
            already_sampled[hoi] +=1
            all_boxes_h.append(boxes_h[i])
            all_boxes_o.append(boxes_o[i])
            all_hois.append(hoi)
            all_objects.append(objects[i])
            all_verbs.append(verbs[i])
            pair_with_original.append(i)
            add_true = True
        else:
            pass
    if add_true:
        all_list.append(dict(
            original_idx=k,
            boxes_h=all_boxes_h,
            boxes_o=all_boxes_o,
            hoi=all_hois,
            object=all_objects,
            verb=all_verbs,
            pair_with_original=pair_with_original,
            filename=filename,
            size=size))
    if sum(already_sampled) == sample_lens * 600:break
print("generate few shot:{}, all images:{}, all pairs{} ".format(sample_lens, len(all_list), sum(already_sampled)))        

# generate vit16 pickle 
file1 = 'union_embeddings_cachemodel_crop_padding_zeros_vitb16.p'
anno_vit16 = pickle.load(open(file1,'rb'))
filenames_vit16 = list(anno_vit16.keys())
new_anno_file = {}
for anno_few in all_list:
#     print(anno_few)
    file_n = anno_few['filename']
    new_anno_file[file_n] = {}
    pair_with_original = anno_few['pair_with_original']
    
#     print(anno_vit16[file_n].keys(),file_n)
    for k in anno_vit16[file_n].keys():
#         print(anno_vit16[file_n][k])
        if k!= 'global_feature':
            new_anno_file[file_n][k] = anno_vit16[file_n][k][pair_with_original]
        else:
            new_anno_file[file_n][k] = anno_vit16[file_n][k]
#     break
assert len(new_anno_file) == len(all_list)

# save pickle
with open(args.save_dir+'few_shot_{}_{}_{}_annofile.p'.format(sample_lens,sum(already_sampled),len(all_list)),'wb') as f:
    pickle.dump(all_list,f)

with open(args.save_dir+'few_shot_{}_{}_{}_vit16emb.p'.format(sample_lens,sum(already_sampled),len(all_list)),'wb') as f:
    pickle.dump(new_anno_file,f)
