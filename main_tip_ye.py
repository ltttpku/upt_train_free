"""
Utilities for training, testing and caching results
for HICO-DET and V-COCO evaluations.

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler


# from upt_tip_cache_model_free import build_detector
from upt_tip_cache_model_free_ye import build_detector

from utils_tip_cache_and_union_ye import custom_collate, CustomisedDLE, DataFactory
import pdb, json
warnings.filterwarnings("ignore")

def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)

    trainset = DataFactory(name=args.dataset, partition=args.partitions[0], data_root=args.data_root)
    testset = DataFactory(name=args.dataset, partition=args.partitions[1], data_root=args.data_root)

    train_loader = DataLoader(
        dataset=trainset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        sampler=DistributedSampler(
            trainset, 
            num_replicas=args.world_size, 
            rank=rank)
    )
    test_loader = DataLoader(
        dataset=testset,
        collate_fn=custom_collate, batch_size=1,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        sampler=torch.utils.data.SequentialSampler(testset)
    )
    # test_loader = DataLoader(
    #     dataset=trainset,
    #     collate_fn=custom_collate, batch_size=1,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=False,
    #     sampler=torch.utils.data.SequentialSampler(trainset)
    # )
    args.human_idx = 0
    if args.dataset == 'hicodet':
        # object_to_target = train_loader.dataset.dataset.object_to_verb
        # args.num_classes = 117
        object_to_target = train_loader.dataset.dataset.object_to_interaction
        args.num_classes = 600
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_classes = 24
    # pdb.set_trace()
    upt = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        upt.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: start from a randomly initialised model")

    engine = CustomisedDLE(
        upt, train_loader,
        max_norm=args.clip_max_norm,
        num_classes=args.num_classes,
        print_interval=args.print_interval,
        find_unused_parameters=True,
        cache_dir=args.output_dir,
        test_loader=test_loader,
        anno_interaction=trainset.dataset.anno_interaction
    )

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            raise NotImplementedError(f"Evaluation on V-COCO has not been implemented.")
        ap = engine.test_hico(test_loader)
        # Fetch indices for rare and non-rare classes
        num_anno = torch.as_tensor(trainset.dataset.anno_interaction)
        rare = torch.nonzero(num_anno < 10).squeeze(1)
        non_rare = torch.nonzero(num_anno >= 10).squeeze(1)
        print(
            f"The mAP is {ap.mean():.4f},"
            f" rare: {ap[rare].mean():.4f},"
            f" none-rare: {ap[non_rare].mean():.4f},"
            
        )
        print(args.resume)
        import datetime
        with open(f'logs/outliers={args.use_outliers}_branch={args.branch}.log', 'a') as f:
            ## write all args
            f.write(str(datetime.datetime.now()))
            f.write('\n')
            json.dump(args.__dict__, f)
            f.write('\n')
            f.write(f'{100 * ap.mean():.2f} ')
            f.write(f'{100 * ap[rare].mean():.2f} ')
            f.write(f'{100 * ap[non_rare].mean():.2f}\n')
            f.write('\n')
        return

    for p in upt.detector.parameters():
        p.requires_grad = False
    for n, p in upt.named_parameters():
        if n.startswith('adapter'):
            p.requires_grad = True
        else:
            p.requires_grad = False
    # for n, p in upt.clip_head.named_parameters():
    for n, p in upt.clip_model.named_parameters():
        

        if n.startswith('visual.positional_embedding') or n.startswith('visual.ln_post') or n.startswith('visual.proj') : 
            # pdb.set_trace()
            p.requires_grad = True
        else: p.requires_grad = False
    
    param_dicts = [{
        "params": [p for n, p in upt.named_parameters()
        if p.requires_grad]
    }]
    # print(param_dicts)
    n_parameters = sum(p.numel() for p in upt.parameters() if p.requires_grad)
    # print()

    print('number of params:', n_parameters)
    # pdb.set_trace()
    # if os.path.exists(args.resume):
    #     print(f"=> Rank {rank}: continue from saved checkpoint {args.resume}")
    #     checkpoint = torch.load(args.resume, map_location='cpu')
    #     # upt.load_state_dict(checkpoint['model_state_dict'])
    #     optim = checkpoint['optim_state_dict']

    # else:
    #     print(f"=> Rank {rank}: start from a randomly initialised model")
        
    optim = torch.optim.AdamW(
        param_dicts, lr=args.lr_head,
        weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop)
    if args.resume:
        
        optim.load_state_dict(checkpoint['optim_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch=checkpoint['epoch']
        iteration = checkpoint['iteration']
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    # Override optimiser and learning rate scheduler
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler, epoch=epoch,iteration=iteration, scaler=scaler)
    else:
        engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)
   

    engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.human_idx = 0; args.num_classes = 117
    object_to_target = dataset.dataset.object_to_verb
    upt = build_detector(args, object_to_target)
    if args.eval:
        upt.eval()

    image, target = dataset[0]
    outputs = upt([image], [target])

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-head', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)

    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--data-root', default='./hicodet')

    # training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--port', default='1233', type=str)
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')
    parser.add_argument('--print-interval', default=500, type=int)
    parser.add_argument('--world-size', default=1, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--visual_mode', default='vit', type=str)
    # add CLIP model resenet 
    parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    parser.add_argument('--clip_visual_width', default=64, type=int)
    parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    parser.add_argument('--clip_text_context_length', default=13, type=int)

    # add CLIP vision
    # parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-16.pt', type=str)
    parser.add_argument('--clip_dir_vit', default='./checkpoints/pretrained_clip/ViT-B-32.pt', type=str)
    parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=32, type=int)
    parser.add_argument('--clip_text_output_dim_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    parser.add_argument('--clip_text_context_length_vit', default=13, type=int)

    parser.add_argument('--topk', default=250, type=int)
    parser.add_argument('--branch', type=str)
    parser.add_argument('--post_process', default=False, action='store_true')
    parser.add_argument('--use_outliers', default=False, action='store_true')
    parser.add_argument('--use_less_confident', default=False, action='store_true')
    parser.add_argument('--num_shot', default=1, type=int)

    parser.add_argument('--neighours_descending', default=False, action='store_true')
    parser.add_argument('--topk_descending', default=False, action='store_true')

    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    # mp.spawn(main, nprocs=args.world_size, args=(args,))
    if args.world_size==1:
        main(0,args)
    else:
        mp.spawn(main, nprocs=args.world_size, args=(args,))
