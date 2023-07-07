import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import resnet

from eva import KFAC as Eva
# from torch_lr_finder import LRFinder
from lr_finder import LRFinder
from fastai.vision.all import Learner
from torch.optim.lr_scheduler import CyclicLR, OneCycleLR
from lion_pytorch import Lion

from sophia import SophiaG

import wandb

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
# add eva
parser.add_argument('--eva', default=True, type=bool,
                    help='Use the eva for preconditioning',
                    )
parser.add_argument('--lrfinder', default=False, type=bool,)
parser.add_argument('--clr', default=False, type=bool,)
parser.add_argument('--oneclr', default=True, type=bool,)
parser.add_argument('--lion', default=False, type=bool,
                    help='Use the lion optimizer',
                    )
parser.add_argument('--sophia', default=False, type=bool,)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()

    wandb.init(config=args,project='eva_j',name=args.arch+'eva'+"lr0.1")

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = torch.nn.DataParallel(resnet.__dict__[args.arch]())
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        criterion.half()
    
    if args.lion:
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.sophia:
        optimizer = SophiaG(model.parameters(), lr=args.lr, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-1)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    
    print(optimizer)

    if args.eva:
        preconditioner = Eva(model,)
                            # lr=0.1,
                            # damping=0.03,
                            # fac_update_freq=1,
                            # kfac_update_freq=1,
                            # kfac_batch_size=16,
                            # kl_clip=0.001,
                            # factor_decay=0.95,
                            # exclude_vocabulary_size=None,
                            # hook_enabled=True,
                            # exclude_parts='')

    # 
    # if args.lrfinder:
    #     lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    #     lr_finder.range_test(train_loader=train_loader, end_lr=0.5, num_iter=100, step_mode="linear")
    #     lr_finder.plot()
    #     lr_finder.reset()
    # elif args.clr:
    #     lr_scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,)
    # else:
    #     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 150], last_epoch=args.start_epoch - 1)

    if args.lrfinder:
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(train_loader=train_loader, start_lr = 0.001, end_lr=0.5, num_iter=100, step_mode="linear")
        lr = lr_finder.plot(log_lr=False, suggest_lr=True)
        # print(lr)
        lr_finder.reset()

    if args.clr is False:
        if args.oneclr is False:
            if args.lion:
                lr_scheduler = CyclicLR(optimizer, base_lr=lr * 0.25, max_lr=lr, cycle_momentum=False) 
            else:
                lr_scheduler = CyclicLR(optimizer, base_lr=lr * 0.25, max_lr=lr, cycle_momentum=True)
        else: 
            if args.lion:
                lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum=False)
            else:
                lr_scheduler = OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, cycle_momentum=True)
    else:
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=args.start_epoch - 1)
        lr_scheduler = None

    # if args.arch in ['resnet1202', 'resnet110']:
    #     # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
    #     # then switch back. In this setup it will correspond for first epoch.
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr*0.1


    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):    
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        
        start_time = time.time()
        top1 = train(train_loader, model, criterion, optimizer, epoch, preconditioner, lr_scheduler)
        # lr_scheduler.step()
        end_time = time.time()
        epoch_time = end_time - start_time

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

        wandb.log({'epoch':epoch,'train_acc':top1,'val_acc':prec1,'lr':optimizer.param_groups[0]['lr'],"epoch_time": epoch_time})
        # 记录"wallclocktime"
        # wandb.log({"wallclocktime": current_time}, step=epoch)

def train(train_loader, model, criterion, optimizer, epoch, preconditioner=None, lr_scheduler=None):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # eva preconditioner
        if args.eva:
            preconditioner.step()

        optimizer.step()
        if lr_scheduler:
            lr_scheduler.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
            
    return top1.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
