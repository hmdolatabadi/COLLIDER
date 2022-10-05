import argparse
import os
import random
import time
import warnings
import sys
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models
import copy

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
from fl_cifar import FacilityLocationCIFAR
from lazyGreedy import lazy_greedy_heap
from utils import *
from data_aux import get_dataset
from models.resnet import resnet18

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'svhn', 'imagenet12'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=['resnet18'] + model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet32)')
parser.add_argument('--exp-str', default='0', type=str, help='number to indicate which experiment it is')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--root_log',type=str, default='final_log')
parser.add_argument('--root_model', type=str, default='final_checkpoint')
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--prefix', type=str, default='LID',
                    help='Prefix for the name of the folder.')
parser.add_argument('--enable_coresets', action='store_true',
                    help="Whether to use coreset selection for training.")
parser.add_argument('--r', type=float, default=2.0,
                    help="Distance threshold (i.e. radius) in calculating clusters.")
parser.add_argument('--fl-ratio', type=float, default=0.8,
                    help="Ratio for number of facilities.")
parser.add_argument('--lid-lambda', type=float, default=0.1,
                    help="Coefficient for adding the LID")

# TODO: create a better data poisoning pipeline than what is currently used (saving the entire poisoned data and loading it on memory)
parser.add_argument('--backdoor', default='badnets', type=str, choices=['no_backdoor', 'htba', 'badnets', 'sig', 'cl', 'wanet'],
                    help='Backdoor attack type. To initialize please refer to ``data_poisoning.ipynb`` and ``data_aux.py``!')
parser.add_argument('--lid_batch_size', default=100, type=int,
                    help='The number of samples in each batch during LID computation!')
parser.add_argument('--injection_rate', type=float, default=0.2,
                    help='Injection rate of poisoned samples into the target class.')
parser.add_argument('--target_class', type=int, default=0,
                    help='The attacker`s target class.')
parser.add_argument('--lid_start_epoch', type=int, default=30,
                    help='The epoch on which we start adding the LID regularizer.')
parser.add_argument('--data_seed', type=int, default=0,
                    help='The data loader seed.')
parser.add_argument('--lid_hist', type=int, default=10,
                    help='The window (number of epochs) for LID moving average.')
parser.add_argument('--lid_overlap', type=int, default=90,
                    help='The number of nearest neighbors used for MLE in LID computation.')

best_acc1 = 0


def main():

    # Initialize the logging and checkpoint saving locations
    args = parser.parse_args()
    if args.enable_coresets and args.backdoor != 'no_backdoor':
        args.store_name = '_'.join([args.prefix, args.dataset, args.arch, args.backdoor, str(args.injection_rate), str(args.fl_ratio), str(args.r), args.exp_str])
    else:
        args.store_name = '_'.join([args.prefix, args.dataset, args.arch, args.backdoor, str(args.fl_ratio), args.exp_str])

    prepare_folders(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'imagenet12':
        args.num_classes = 12
    elif args.dataset == 'gtsrb':
        args.num_classes = 13
    else:
        args.num_classes = 10

    if args.arch != 'resnet18':
        model = models.__dict__[args.arch](num_classes=args.num_classes)
    else:
        model = resnet18(pretrained=False, num_classes=args.num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:

                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Get the poisoned training dataset, a validation set for model selection,
    # as well as the poisoned and clean test sets. 
    data_root     = args.data_root
    train_dataset = get_dataset(root=data_root,
                                dataset=args.dataset,
                                attack_type=args.backdoor,
                                injection_rate=args.injection_rate,
                                data_transform='train',
                                partition='train',
                                valid_frac=0.04,
                                lid_batch_size=args.lid_batch_size,
                                indexed=True,
                                seed=args.data_seed)

    val_dataset = get_dataset(root=data_root,
                              dataset=args.dataset,
                              attack_type='no_backdoor',
                              injection_rate=args.injection_rate,
                              data_transform='test',
                              partition='val',
                              valid_frac=0.04,
                              lid_batch_size=args.lid_batch_size,
                              indexed=False,
                              seed=args.data_seed)

    testbd_dataset = get_dataset(root=data_root,
                                 dataset=args.dataset,
                                 attack_type=args.backdoor,
                                 injection_rate=args.injection_rate,
                                 data_transform='test',
                                 partition='test',
                                 valid_frac=0.04,
                                 lid_batch_size=args.lid_batch_size,
                                 indexed=False,
                                 seed=args.data_seed)

    test_dataset   = get_dataset(root=data_root,
                                 dataset=args.dataset,
                                 attack_type='no_backdoor',
                                 injection_rate=args.injection_rate,
                                 data_transform='test',
                                 partition='test',
                                 valid_frac=0.04,
                                 lid_batch_size=args.lid_batch_size,
                                 indexed=False,
                                 seed=args.data_seed)
    
    # the training criterion
    criterion = nn.CrossEntropyLoss(reduction='none').cuda(args.gpu)
    
    # setting the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    trainval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.lid_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    testbd_loader = torch.utils.data.DataLoader(
        testbd_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # in case we need to evaluate a pre-trained model, just set the --evaluate argument
    if args.evaluate:
        validate(test_loader, model, criterion, 0, args)
        validate(testbd_loader, model, criterion, 0, args, bd=True)
        return

    # setting the LR scheduler
    if  args.dataset == 'imagenet12':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[72, 144], last_epoch=args.start_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[80, 100], last_epoch=args.start_epoch - 1)

    # init log for training
    log_training = open(os.path.join(args.root_log, args.store_name, 'log.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    tf_writer = SummaryWriter(log_dir=os.path.join(args.root_log, args.store_name))
    weights   = [1] * len(train_dataset)
    weights   = torch.FloatTensor(weights)

    # init the LID variables in case we use it during the training
    if args.lid_start_epoch < args.epochs:
        LID_history = np.zeros((train_dataset.tensors[0].shape[0], args.lid_hist))
        n_hist      = 0
        removed_ind = []
        remove_rate = (1 - args.fl_ratio)/(args.epochs - args.lid_start_epoch)

    total_time = []
    for epoch in range(args.start_epoch, args.epochs):

        start = time.time()
        subset_selection_time = 0

        if args.enable_coresets and epoch >= 0:

            train_dataset.switch_data()

            # last layer gradient estimation, saving the features for LID computation
            grads_all, labels, feats = estimate_grads(trainval_loader, model, criterion, args, epoch, log_training)

            # using the evaluated features for LID computation and taking the moving average
            if epoch >= args.lid_start_epoch:
                LIDs_all           = np.abs(np.asarray(get_LID(feats, model, args.lid_overlap)[0]))
                LID_history        = np.roll(LID_history, axis=1, shift=-1)
                LID_history[:, -1] = LIDs_all.ravel()

                if n_hist < args.lid_hist:
                    n_hist += 1

                LIDs_all = np.sum(LID_history, axis=1, keepdims=True)/n_hist

            # init
            ssets        = []
            weights      = []
            LID_by_class = {}

            # class-wise coreset selection
            for c in range(args.num_classes):
                sample_ids = np.where((labels == c) == True)[0]
                cls_len    = sample_ids.shape[0]

                # getting the LID, and removing the ones with the highest value permanently
                if epoch >= args.lid_start_epoch:
                    sample_ids   = np.setdiff1d(sample_ids, np.array(removed_ind))
                    top_K        = int(np.floor(remove_rate * cls_len))
                    LIDs         = LIDs_all[sample_ids]
                    max_LID_id   = LIDs.ravel().argsort()[-top_K:][::-1]
                    removed_ind += list(sample_ids[max_LID_id])
                    sample_ids   = np.setdiff1d(sample_ids, sample_ids[max_LID_id])
                    LIDs         = LIDs_all[sample_ids]
                    LID_by_class[str(c)] = LIDs

                # computing d_{i,j} from Eq. (5)
                grads = grads_all[sample_ids]
                dists = pairwise_distances(grads)
                
                # adding the LID regularizer
                if epoch >= args.lid_start_epoch:
                    dists += args.lid_lambda * LIDs.T

                # setting up the facility location (FL) problem and solving it with greedy selection
                weight = np.sum(dists < args.r, axis=1)
                V = range(len(grads))
                F = FacilityLocationCIFAR(V, D=dists)
                B = min(int(args.fl_ratio * cls_len), int(len(grads)))
                sset, vals = lazy_greedy_heap(F, V, B)
                weights.extend(weight[sset].tolist())
                sset = sample_ids[np.array(sset)]
                ssets += list(sset)

            # set the train_loader to the data in the coreset
            subset_selection_time = time.time() - start
            weights = torch.FloatTensor(weights)
            train_dataset.adjust_base_indx_tmp(ssets)
            label_acc = train_dataset.estimate_label_acc(ssets)
            tf_writer.add_scalar('label_acc', label_acc, epoch)
            log_training.write('epoch %d label acc: %f\n'%(epoch, label_acc))
            print("change train loader")

        # train for one epoch
        start = time.time()

        if args.enable_coresets and epoch > 0:
            train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=True)
        else:
            train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=False)

        train_time = time.time() - start
        total_time.append(train_time + subset_selection_time)

        # evaluate on validation set
        acc1    = validate(val_loader, model, criterion, epoch, args, log_training, tf_writer)
        acc1_bd = validate(testbd_loader, model, criterion, epoch, args, log_training, tf_writer, bd=True)

        # remember best acc@1 and save checkpoint
        is_best   = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        tf_writer.add_scalar('acc/test_top1_best', best_acc1, epoch)
        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_model_acc1_bd': acc1_bd,
            'optimizer' : optimizer.state_dict(),
        }, is_best)
        lr_scheduler.step()

    print('best_acc1: {:.4f}'.format(best_acc1.item()))
    output_bd = 'Total Train Time: %.3f\n' % (np.array(total_time).sum()/3600.)
    print(output_bd)


def train(train_loader, model, criterion, weights, optimizer, epoch, args, log_training, tf_writer, fetch=False):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time  = AverageMeter('Data', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')
    top1       = AverageMeter('Acc@1', ':6.2f')
    top5       = AverageMeter('Acc@5', ':6.2f')
    progress   = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                               top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        input, target, index = batch
        if fetch:
            input_b = train_loader.dataset.fetch(target)[0]
            lam     = np.random.beta(1, 0.1)
            input   = lam * input + (1 - lam) * input_b
        c_weights = weights[index]
        c_weights = c_weights.type(torch.FloatTensor)
        c_weights = c_weights / c_weights.sum()
        if args.gpu is not None:
            c_weights = c_weights.to(args.gpu, non_blocking=True)
    
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.type(torch.FloatTensor)
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output, feats = model(input)
        loss = criterion(output, target)
        loss = (loss * c_weights).sum()
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    tf_writer.add_scalar('loss/train', losses.avg, epoch)
    tf_writer.add_scalar('acc/train_top1', top1.avg, epoch)
    tf_writer.add_scalar('acc/train_top5', top5.avg, epoch)
    tf_writer.add_scalar('lr', optimizer.param_groups[-1]['lr'], epoch)

def validate(val_loader, model, criterion, epoch, args, log_training=None, tf_writer=None, bd=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses     = AverageMeter('Loss', ':.4e')
    top1       = AverageMeter('Acc@1', ':6.2f')
    top5       = AverageMeter('Acc@5', ':6.2f')
    progress   = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                               prefix='Validate: ' if not bd else 'Backdoor Acc:')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.type(torch.FloatTensor)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output, feats = model(input)
            loss = criterion(output, target)
            loss = loss.mean()

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        if tf_writer is not None:

            if not bd:
                tf_writer.add_scalar('loss/test', losses.avg, epoch)
                tf_writer.add_scalar('acc/test_top1', top1.avg, epoch)
                tf_writer.add_scalar('acc/test_top5', top5.avg, epoch)
                log_training.write('epoch %d val acc: %f\n'%(epoch, top1.avg))
            else:
                tf_writer.add_scalar('loss/test_bd', losses.avg, epoch)
                tf_writer.add_scalar('acc/test_bd_top1', top1.avg, epoch)
                tf_writer.add_scalar('acc/test_bd_top5', top5.avg, epoch)
                log_training.write('epoch %d bd acc: %f\n'%(epoch, top1.avg))

    return top1.avg


def estimate_grads(trainval_loader, model, criterion, args, epoch, log_training):
    # switch to train mode
    model.train()

    all_grads   = []
    all_targets = []
    all_feats   = []
    top1        = AverageMeter('Acc@1', ':6.2f')

    for i, (input, target, idx) in enumerate(trainval_loader):
        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)

        all_targets.append(target)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        feat = model(input, freeze=True)
        feat.requires_grad_(True)
        output = model.fc(feat)

        _, pred = torch.max(output, 1)
        loss = criterion(output, target).mean()
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        est_grad = grad(loss, feat)
        all_grads.append(est_grad[0].detach().cpu().numpy())
        all_feats.append(feat.detach().cpu().numpy())

    all_grads   = np.vstack(all_grads)
    all_targets = np.hstack(all_targets)
    all_feats   = np.vstack(all_feats)
    log_training.write('epoch %d train acc: %f\n'%(epoch, top1.avg))

    return all_grads, all_targets, all_feats


def get_LID(feats, overlap=90, batch_size=100):
    '''
    Compute LID score on the whole data
    return: LID score
    '''

    LID          = []
    overlap_list = [overlap]
    num_output   = 1

    for _ in overlap_list:
        LID.append([])

    with torch.no_grad():

        feats       = feats.astype(np.float32).reshape((feats.shape[0], -1))
        total_batch = feats.shape[0]
        step_size   = total_batch // batch_size

        for idx in range(step_size):
            begin  = idx * batch_size
            end    = idx * batch_size + batch_size
            X_act  = [feats[begin: end]]

            # LID
            list_counter = 0
            for overlap in overlap_list:
                LID_list = []

                for j in range(num_output):
                    lid_score = mle_batch(X_act[j], X_act[j], k=overlap)
                    lid_score = lid_score.reshape((lid_score.shape[0], -1))

                    LID_list.append(lid_score)


                LID_concat = LID_list[0]

                for i in range(1, num_output):
                    LID_concat = np.concatenate((LID_concat, LID_list[i]), axis=1)

                LID[list_counter].extend(LID_concat)
                list_counter += 1

    return LID

if __name__ == '__main__':
    main()