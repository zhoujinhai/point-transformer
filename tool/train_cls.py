
import os
import time
import random
import numpy as np
import logging
import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from util import config
from util.s3dis import S3DIS
from util.SupportDataLoader import SemSegSupportDataset
from util.toothDataLoader import SemSegToothDataset, ClsToothDataset
from util.common_util import AverageMeter, intersectionAndUnionGPU, find_free_port
from util.data_util import collate_fn, collate_fn_cls
from util import transform as t


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.data_name == 's3dis':
        S3DIS(split='train', data_root=args.data_root, test_area=args.test_area)
        S3DIS(split='val', data_root=args.data_root, test_area=args.test_area)
    elif args.data_name == 'support':
        print("Check train and val Data!")
        #SemSegSupportDataset(split='train', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim)
        #SemSegSupportDataset(split='val', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim)
    elif args.data_name == 'teeth':
        print("Check train and val Data!")
        #SemSegToothDataset(split='train', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim)
        #SemSegToothDataset(split='val', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim)
    else:
        raise NotImplementedError()
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args, best_iou
    args, best_iou = argss, 0
    best_mAcc = 0
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    if args.arch == "pointtransformer_cls_small_repro":
        from model.pointtransformer.pointtransformer_seg import pointtransformer_cls_small_repro as Model 
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(in_channels=args.fea_dim, n_cls=args.classes)
    if args.sync_bn:
       model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label).cuda()
    if args.optimizer == "Adam":
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        # logger.info(model)
    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "transformer" in args.arch else False
        )

    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume)) 
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            # torch.save(checkpoint, args.save_path + '/model/model_best_copy.pth') 
            args.start_epoch = checkpoint['epoch']
            # -------Load others weight------- 
            if args.other_weight:
                simplified_state_dict = model.state_dict() 
                filtered_state_dict = {k: v for k, v in checkpoint.items() 
                                    if k in simplified_state_dict and v.size() == simplified_state_dict[k].size()}
                model.load_state_dict(filtered_state_dict, strict=False)
                if 'optimizer' in checkpoint:
                    optimizer_state = checkpoint['optimizer']
                    
                    for i, group in enumerate(optimizer.param_groups):
                        if i < len(optimizer_state['param_groups']):
                            orig_group = optimizer_state['param_groups'][i] 
                            for key in ['lr', 'weight_decay', 'momentum', 'betas', 'eps']:
                                if key in orig_group:
                                    group[key] = orig_group[key]
                    
                    print("Optimizer hyperparameters loaded")
            # -------End others weights--------
            else:
                model.load_state_dict(checkpoint['state_dict'], strict=True) # True
                if "optimizer" in checkpoint.keys():
                    optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            # ---------------

            best_iou = 0.0
            # best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    train_transform = t.Compose([t.RandomScale([0.9, 1.1]), t.ChromaticAutoContrast(), t.ChromaticTranslation(), t.ChromaticJitter(), t.HueSaturationTranslation()])
    if args.data_name == 'teeth':
        train_data = ClsToothDataset(split='all', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim, b_limit_point=args.limit_point) 
        weights = torch.Tensor(train_data.label_weights).cuda()
        dataset_size = len(train_data)
        train_size = int(dataset_size * 0.8)  # 80% 璁粌
        test_size = dataset_size - train_size  # 20% 娴嬭瘯
        torch.manual_seed(42)

        # 闅忔満鍒掑垎鏁版嵁闆?        train_data, val_data = torch.utils.data.random_split(
            train_data, 
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # 淇濊瘉鍙噸澶嶆€?        )
        
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=args.ignore_label).cuda() 
    else:
        raise NotImplementedError()
 
    if main_process():
            logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn_cls)

    val_loader = None
    if args.evaluate:
        # val_transform = None
        # if args.data_name == 'teeth':
        #     val_data = ClsToothDataset(split='val', root=args.data_root, npoints=args.npoints, n_class=args.classes, f_cols=args.fea_dim, b_limit_point=args.limit_point)  
        # else:
        #     raise NotImplementedError()
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn_cls)

    best_acc = 0.0
    best_mean_class_acc = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        logger.info('====>Epoch: {:d}'.format(epoch))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        loss_train, all_acc_train, mean_class_acc_train = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('acc_train', all_acc_train, epoch_log)  
            writer.add_scalar('mean_class_acc_train', mean_class_acc_train, epoch_log)

        is_best = False
        is_best_mAcc = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, all_acc_val, mean_class_acc_val = validate(val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('acc_val', all_acc_val, epoch_log)   
                writer.add_scalar('mean_class_acc_val', mean_class_acc_val, epoch_log)
                is_best = all_acc_val > best_acc
                best_acc = max(best_acc, all_acc_val)
                 
                is_best_mean_class_acc = mean_class_acc_val > best_mean_class_acc
                best_mean_class_acc = max(mean_class_acc_val, best_mean_class_acc)

        if (epoch_log % args.save_freq == 0) and main_process() or is_best or is_best_mean_class_acc:
            filename = args.save_path + '/model/model_last.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_acc': float(best_acc), 'is_best': bool(is_best)}, filename)
            if is_best:
                logger.info('Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path + '/model/model_best.pth')
            if is_best_mAcc:
                logger.info('Best validation mAcc updated to: {:.4f}'.format(best_mean_class_acc))
                shutil.copyfile(filename, args.save_path + '/model/model_best_mean_class_acc.pth')

    # eval last
    if args.evaluate:
        if args.data_name == 'shapenet':
            raise NotImplementedError()
        else:
            loss_val, all_acc_val, mean_class_acc_val = validate(val_loader, model, criterion)

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (all_acc_val))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()  # 鏁翠綋鍑嗙‘鐜?    class_acc_meter = AverageMeter()  # 姣忎釜绫诲埆鐨勫噯纭巼
    confusion_matrix = None  # 娣锋穯鐭╅樀

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    
    # 鍒濆鍖栨贩娣嗙煩闃?    if args.classes > 0:
        confusion_matrix = torch.zeros(args.classes, args.classes).cuda()
    
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (b), (b)
        data_time.update(time.time() - end)
        
        # 鏁版嵁杞Щ鍒癎PU
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True) 
        
        # 鍓嶅悜浼犳挱
        output = model([coord, feat, offset])
        
        # 纭繚target鏄?D寮犻噺锛堝垎绫讳换鍔★級
        if target.dim() > 1:
            target = target.squeeze()
        print("target: ", target, "output: ", output)
        # 璁＄畻鎹熷け
        loss = criterion(output, target) 
        
        # 姊害绱Н
        if args.accumulation > 1:
            loss = loss / args.accumulation
            loss.backward()
            if (i + 1) % args.accumulation == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
        
        # 璁＄畻棰勬祴缁撴灉
        _, preds = torch.max(output, 1)
        
        # 璁＄畻鍑嗙‘鐜?        correct = (preds == target).float()
        accuracy = correct.mean()
        
        # 鏇存柊娣锋穯鐭╅樀
        if confusion_matrix is not None:
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        # 鍒嗗竷寮忚缁冨鐞?        n = target.size(0)  # 鎵规澶у皬
        if args.multiprocessing_distributed:
            # 鍚屾鎹熷け
            loss_sum = loss * n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss_sum), dist.all_reduce(count)
            n_total = count.item()
            loss = loss_sum / n_total
            
            # 鍚屾鍑嗙‘鐜?            correct_sum = correct.sum()
            dist.all_reduce(correct_sum)
            accuracy = correct_sum / n_total
            
            # 鍚屾娣锋穯鐭╅樀
            if confusion_matrix is not None:
                dist.all_reduce(confusion_matrix)
        
        # 鏇存柊鎸囨爣
        loss_meter.update(loss.item(), n)
        acc_meter.update(accuracy.item(), n)
        
        # 璁＄畻姣忎釜绫诲埆鐨勫噯纭巼
        if confusion_matrix is not None:
            class_acc = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-10)
            mean_class_acc = class_acc.mean().item()
            class_acc_meter.update(mean_class_acc, n)
        
        batch_time.update(time.time() - end)
        end = time.time()

        # 璁＄畻鍓╀綑鏃堕棿
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        # 鎵撳嵃鏃ュ織
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Acc {acc_meter.val:.4f} '
                        'MeanClassAcc {class_acc_meter.val:.4f}.'.format(
                            epoch+1, args.epochs, i + 1, len(train_loader),
                            batch_time=batch_time, data_time=data_time,
                            remain_time=remain_time,
                            loss_meter=loss_meter,
                            acc_meter=acc_meter,
                            class_acc_meter=class_acc_meter))
        
        # TensorBoard璁板綍
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('acc_train_batch', acc_meter.val, current_iter)
            writer.add_scalar('mean_class_acc_train_batch', class_acc_meter.val, current_iter)
    
    # 璁＄畻鏈€缁堟寚鏍?    if confusion_matrix is not None:
        confusion_matrix = confusion_matrix.cpu().numpy()
        class_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-10)
        mean_class_acc = np.mean(class_acc)
    else:
        mean_class_acc = 0.0
    
    all_acc = acc_meter.avg
    
    if main_process():
        logger.info('Train result at epoch [{}/{}]: '
                    'Loss {loss_avg:.4f} '
                    'Acc {acc_avg:.4f} '
                    'MeanClassAcc {mean_class_acc:.4f}.'.format(
                        epoch+1, args.epochs,
                        loss_avg=loss_meter.avg,
                        acc_avg=all_acc,
                        mean_class_acc=mean_class_acc))
        
        # 璁板綍姣忎釜绫诲埆鐨勫噯纭巼
        if confusion_matrix is not None:
            for i in range(args.classes):
                logger.info('Class {}: Acc {:.4f}'.format(i, class_acc[i]))
    
    torch.cuda.empty_cache()
    return loss_meter.avg, all_acc, mean_class_acc
 
def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()  # 鏁翠綋鍑嗙‘鐜?    class_acc_meter = AverageMeter()  # 姣忎釜绫诲埆鐨勫噯纭巼
    
    # 鍒濆鍖栨贩娣嗙煩闃?    confusion_matrix = torch.zeros(args.classes, args.classes).cuda()
    
    model.eval()
    end = time.time()
    
    with torch.no_grad():
        for i, (coord, feat, target, offset) in enumerate(val_loader):
            data_time.update(time.time() - end)
            
            # 鏁版嵁杞Щ鍒癎PU
            coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
            
            # 纭繚target鏄?D寮犻噺锛堝垎绫讳换鍔★級
            if target.dim() > 1:
                target = target.squeeze()
            
            # 鍓嶅悜浼犳挱
            output = model([coord, feat, offset])
            
            # 璁＄畻鎹熷け
            loss = criterion(output, target)
            
            # 璁＄畻棰勬祴缁撴灉
            _, preds = torch.max(output, 1)
            
            # 璁＄畻鍑嗙‘鐜?            correct = (preds == target).float()
            accuracy = correct.mean()
            
            # 鏇存柊娣锋穯鐭╅樀
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            # 鍒嗗竷寮忚缁冨鐞?            n = target.size(0)
            if args.multiprocessing_distributed:
                # 鍚屾鎹熷け
                loss_sum = loss * n
                count = target.new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss_sum), dist.all_reduce(count)
                n_total = count.item()
                loss = loss_sum / n_total
                
                # 鍚屾鍑嗙‘鐜?                correct_sum = correct.sum()
                dist.all_reduce(correct_sum)
                accuracy = correct_sum / n_total
                
                # 鍚屾娣锋穯鐭╅樀
                dist.all_reduce(confusion_matrix)
            
            # 璁＄畻姣忎釜绫诲埆鐨勫噯纭巼
            class_acc = confusion_matrix.diag() / (confusion_matrix.sum(1) + 1e-10)
            mean_class_acc = class_acc.mean().item()
            
            # 鏇存柊鎸囨爣
            loss_meter.update(loss.item(), n)
            acc_meter.update(accuracy.item(), n)
            class_acc_meter.update(mean_class_acc, n)
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # 鎵撳嵃杩涘害
            if (i + 1) % args.print_freq == 0 and main_process():
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Acc {acc_meter.val:.4f} '
                            'MeanClassAcc {class_acc_meter.val:.4f}.'.format(
                                i + 1, len(val_loader),
                                data_time=data_time,
                                batch_time=batch_time,
                                loss_meter=loss_meter,
                                acc_meter=acc_meter,
                                class_acc_meter=class_acc_meter))
    
    # 璁＄畻鏈€缁堟寚鏍?    confusion_matrix = confusion_matrix.cpu().numpy()
    class_acc = np.diag(confusion_matrix) / (confusion_matrix.sum(1) + 1e-10)
    mean_class_acc = np.mean(class_acc)
    all_acc = acc_meter.avg
    
    if main_process():
        logger.info('Val result: '
                    'Loss {loss_avg:.4f} '
                    'Acc {acc_avg:.4f} '
                    'MeanClassAcc {mean_class_acc:.4f}.'.format(
                        loss_avg=loss_meter.avg,
                        acc_avg=all_acc,
                        mean_class_acc=mean_class_acc))
        
        # 璁板綍姣忎釜绫诲埆鐨勫噯纭巼
        for i in range(args.classes):
            logger.info('Class_{} Result: accuracy {:.4f}.'.format(i, class_acc[i]))
        
        # 鍙€夛細璁板綍娣锋穯鐭╅樀
        if args.classes <= 10:  # 绫诲埆杈冨皯鏃舵墦鍗版贩娣嗙煩闃?            logger.info('Confusion Matrix:')
            logger.info(confusion_matrix)
        
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
    
    return loss_meter.avg, all_acc, mean_class_acc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()
