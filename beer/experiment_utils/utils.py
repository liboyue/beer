import argparse
import os
from time import time
from pprint import pformat
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

import beer
from beer import CommunicationGraph
from beer import log
from beer.utils import all_reduce_tensors


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--optimizer',     default='CHOCO_SGD',     type=str,  help='Optimizer'                       )
    parser.add_argument('--compression_type',     default='unbiased_random',     type=str)
    parser.add_argument('--compression_params',     nargs="+",     type=int)
    parser.add_argument('--graph_type',          default='er', type=str)
    parser.add_argument('--graph_params',           nargs="+", type=float)
    parser.add_argument('--gamma',           default=0, type=float)

    # Training args
    parser.add_argument('--epochs',        default=1,     type=int,            help='Number of epochs'                       )
    parser.add_argument('--batch_size',    default=100,   type=int,            help='Batch size per worker'                  )
    parser.add_argument('--lr',            default=1e-3,  type=float,          help='Learning rate'                          )
    parser.add_argument('--momentum',      default=0,     type=float,          help='Momentum'                               )
    parser.add_argument('--val_interval',  default=None,  type=int,            help='Number of iterations before validation' )
    parser.add_argument('--disp_interval', default=100,   type=int,            help='Number of iterations before display'    )
    parser.add_argument('--num_workers',   default=0,     type=int,            help='Number of workers for data loader'      )
    parser.add_argument('--cpu',           default=False, action='store_true', help='Use CPU'                                )
    parser.add_argument('--deterministic', default=False, action='store_true', help='Use fixed random seed'                  )

    # Distributed args
    parser.add_argument('--backend',             default='nccl',        type=str,             help='Distributed backend',           choices=['nccl', 'gloo', 'mpi'])
    parser.add_argument('--n_peers',             default=None,          type=int,             help='Number of iterations before synchroning'                       )

    # Misc args
    parser.add_argument('--data_path',      default=None,   type=str, help='Path to the data folder' )
    parser.add_argument('--output_dir',     default=None,   type=str, help='Path to the output folder' )
    parser.add_argument('-v','--verbosity', default='INFO', type=str, help='Verbosity of log', choices=['DEBUG', 'INFO', 'WARN'] )

    return parser

def parse_args(parser=None):
    if parser is None:
        parser = get_parser()

    args = parser.parse_args()
    
    # Load options from envs
    for name in ['MASTER_ADDR', 'MASTER_PORT']:
        setattr(args, name.lower(), os.environ[name])
    for name in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'WORLD_LOCAL_SIZE', 'WORLD_NODE_RANK']:
        setattr(args, name.lower(), int(os.environ[name]))
    args.node_rank = args.world_node_rank

    return args


def validate_args(args):
    # Check the backend and device compatibility
    if (not args.cpu) and (not torch.cuda.is_available()):
        log.warn('GPU is not availabel, using CPU instead')
        args.cpu = True

    args.device = torch.device('cpu') if args.cpu else torch.device('cuda:%d' % (args.local_rank % torch.cuda.device_count()))

    if args.cpu and args.backend == 'nccl':
        log.warn('Setting backend to gloo when using CPU')
        args.backend = 'gloo'

    return args


def init(args):
    log.set_rank(args.rank)
    if args.output_dir is not None:
        log.set_directory(args.output_dir)
    log.set_level(args.verbosity)

    args = validate_args(args)

    log.info('Configurations:\n' + pformat(args.__dict__))

    log.info('world_size = %d, batch_size = %d, device = %s, backend = %s',
              args.world_size, args.batch_size, args.device, args.backend)

    if not args.cpu:
        # log.info(f'Using device {args.local_rank % torch.cuda.device_count()}')
        # torch.cuda.set_device(int(args.local_rank % torch.cuda.device_count()))
        torch.backends.cudnn.benchmark = True

    if args.deterministic:
        torch.manual_seed(args.rank)
        np.random.seed(args.rank)
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(args.rank)

    dist.init_process_group(args.backend, world_size=args.world_size, rank=args.rank)


def wrap_model(model, args):

    graph = CommunicationGraph(args.world_size, rank=args.rank, graph_type=args.graph_type, graph_params=args.graph_params)
    if args.optimizer == 'CHOCO_SGD':
        model = beer.DataParallel(model)
        optimizer = beer.optim.CHOCO_SGD(model, lr=args.lr, gamma=args.gamma, world_size=args.world_size, rank=args.rank, G=graph, compression_type=args.compression_type, compression_params=args.compression_params)
    elif args.optimizer == 'BEER':
        model = beer.DataParallel(model, use_ref_module=True)
        optimizer = beer.optim.BEER(model, lr=args.lr, gamma=args.gamma, world_size=args.world_size, rank=args.rank, G=graph, compression_type=args.compression_type, compression_params=args.compression_params)
    elif args.optimizer == 'SGD':
        model = beer.DataParallel(model)
        optimizer = beer.optim.SGD(model, lr=args.lr, world_size=args.world_size, rank=args.rank, G=graph, compression_type=args.compression_type, compression_params=args.compression_params)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not implemented")

    return model, optimizer


def accuracy(output, target):
    if type(output) == tuple:
        output = output[0]
    pred = output.data.max(1, keepdim=True)[1]
    return pred.eq(target.data.view_as(pred)).cpu().float().mean()


def train(model, criterion, optimizer, train_loader, args, val_loader=None, train_loader_for_val=None, exp_name=None, classes=None, scheduler=None):

    def _val():
        if args.val_interval is not None:
            val_start = time()
            model.eval()
            if args.rank == 0:
                val_res.append([i, train_time, run_time, *validate(model, val_loader, criterion, classes=classes, device=args.device)])
            model.train()
            val_end = time()
            return val_end - val_start
        else:
            return 0

    def _save():
        if args.rank == 0:
            fname = get_fname(args, exp_name=None)
            save_data(train_res, val_res, fname, output_dir=args.output_dir)
            log.debug('Data saved to %s', fname)

    def _eta():
        _time = train_time / i * (total_batches - i)
        if args.val_interval is not None:
            _time += val_time / (i // args.val_interval + 1) * ((total_batches - i) // args.val_interval + 1)

        h = _time / 3600
        if h > 1:
            return "%.2fh" % h

        m = _time / 60
        if m > 1:
            return "%.2fm" % m

        return "%.2fs" % _time

    total_batches = len(train_loader) * args.epochs
    train_res = []
    val_res = []
    # running_loss = []
    # running_acc = []
    i = 0
    val_time = run_time = train_time = 0
    train_start = time()
    printed = False

    val_time += _val()

    log.info('Training started')
    model.train()
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):

        for _, (data, target) in enumerate(train_loader):

            i += 1

            target = target.to(device=args.device, non_blocking=True)
            data = data.to(device=args.device, non_blocking=True)

            # ==== Step begin ====
            output = model(data)
            if type(output) == tuple:
                # log.info('Tuple loss!')
                loss = criterion(output[0], target)
                loss.backward()
                loss_ref = criterion(output[1], target)
                loss_ref.backward()
            else:
                loss = criterion(output, target)
                loss.backward()

            log.debug('[%d/%d, %5d/%d] optimizer step', epoch, args.epochs, i, total_batches)
            optimizer.step()
            optimizer.zero_grad()

            # grad_norm = sum([t.grad.norm(2) ** 2 for t in model.module.parameters()])
            # dist.all_reduce(grad_norm)
            # grad_norm = np.sqrt(grad_norm.item() / dist.get_world_size())

            # dist.all_reduce(loss)
            # loss = loss.item() / dist.get_world_size()

            # running_loss.append(loss)
            # if classes is not None:
                # acc = accuracy(output, target).item()
                # running_acc.append(acc)

            log.debug('Step done')
            # ==== Step done ====

            current_time = time()
            run_time = current_time - train_start
            train_time = run_time - val_time

            model.eval()
            grad_norm, loss, acc = validate(model, train_loader_for_val, criterion, classes=classes, device=args.device, distributed=True, show_log=False)
            model.train()

            tmp_res = [i, train_time, run_time, grad_norm, loss]
            if classes is not None:
                tmp_res += [acc]

            train_res.append(tmp_res)

            if i % args.disp_interval == 0:
                log.info('[%d/%d, %5d/%d] local running loss %.5f, local running acc %.5f%%, average train time %.4f seconds per batch, eta %s',
                        epoch, args.epochs, i, total_batches, loss, acc * 100, train_time / i, _eta())
                        # epoch, args.epochs, i, total_batches, np.mean(running_loss), np.mean(running_acc) * 100, train_time / i, _eta())
                _save()
                # running_loss = []
                # running_acc = []

            if args.val_interval is not None and i % args.val_interval == 0:
                val_time += _val()
                # Update saved data after every validation
                _save()

            # end for

        current_time = time()
        run_time = current_time - train_start
        train_time = run_time - val_time

        log.info('Training epoch %d ends, total run time %.4f seconds, average train time %.4f seconds per batch', epoch, run_time, train_time / i)

        if scheduler is not None:
            log.debug('schedule.step() called')
            scheduler.step()
     

    if args.val_interval is not None and i % args.val_interval != 0:
        val_time += _val()

    current_time = time()
    run_time = current_time - train_start
    train_time = run_time - val_time

    _save()

    if args.rank == 0:
        if classes is not None:
            best_acc = max([x[-1] for x in val_res])
            log.info('Training finished, %d epochs, final val loss %.5f, final val acc %.5f%%, best val acc %.5f%%',
                     epoch, val_res[-1][-2], val_res[-1][-1] * 100, best_acc * 100)
        else:
            log.info('Training finished, %d epochs, final val loss %.5f', epoch, val_res[-1][-1])

    return train_res, val_res


def validate(model, val_loader, criterion, classes=None, device=None, distributed=False, show_log=True):

    if show_log:
        _log = log.info
    else:
        _log = log.debug

    _log('Validating model')

    model.zero_grad()
    loss = 0

    if classes is not None:
        confusion_matrix = torch.zeros((len(classes), len(classes)), device=device)

    for data, target in val_loader:
        target = target.to(device=device, non_blocking=True)
        data = data.to(device=device, non_blocking=True)
        output = model(data)

        loss += criterion(output, target)

        if classes is not None:
            _, predicted = torch.max(output, 1)
            for i in range(len(target)):
                l = target[i]
                p = predicted[i]
                confusion_matrix[l][p] += 1

    loss /= len(val_loader)
    loss.backward()

    with torch.no_grad():

        if distributed:
            if classes is not None:
                reqs = all_reduce_tensors([loss, confusion_matrix] + [t.grad for t in model.val_module.parameters()])
            else:
                reqs = all_reduce_tensors([loss] + [t.grad for t in model.val_module.parameters()])
            for req in reqs:
                if req is not None:
                    req.wait()

            world_size = dist.get_world_size()
            loss = loss.cpu().item() / world_size
            grad_norm = torch.sqrt(sum([t.grad.norm(2) ** 2 for t in model.val_module.parameters()])).cpu().item() / world_size
        else:
            loss = loss.cpu().item()
            grad_norm = torch.sqrt(sum([t.grad.norm(2) ** 2 for t in model.val_module.parameters()])).cpu().item()

    if classes is not None:

        confusion_matrix = confusion_matrix.cpu()
        acc = np.diag(confusion_matrix).sum() / confusion_matrix.sum()

        confusion_matrix /= confusion_matrix.sum(axis=1)
        # log.debug(confusion_matrix)

        max_len = str(max([len(str(c)) for c in classes]))
        if len(classes) > 10:
            _log('Accuracy of first 5 classes')
            for i in range(5):
                _log('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])

            _log('Accuracy of last 5 classes')
            for i in range(len(classes) - 5, len(classes)):
                _log('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])
        else:
            _log('Accuracy of each class')
            for i in range(len(classes)):
                _log('%-' + max_len + 's: %8.5f%%', classes[i], 100 * confusion_matrix[i, i])

        _log('Validation loss %.5f, grad norm %.5f, accuracy %.5f%%', loss, grad_norm, acc * 100)

        return grad_norm, loss, acc

    else:
        _log('Validation loss %.5f, grad norm %.5f', loss, grad_norm)
        return grad_norm, loss


def get_fname(args, exp_name=None):

    fname = f'{args.optimizer}_lr_{args.lr}'
    if 'gamma' in args:
        fname += f'_gamma_{args.gamma}'

    fname += f'{args.world_size}_{args.device.type}'

    return fname


def save_data(train_res, val_res, fname, output_dir='data'):

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if output_dir.endswith('/'):
        output_dir = output_dir[:-1]

    header = 'iterations train_time run_time grad_norm loss'
    if len(train_res[0]) == 6:
        header += ' accuracy'

    def _save(res, name):
        res = np.array(res)
        np.savetxt(output_dir + '/full_' + fname + name, res, header=header, comments='')

        # Downsample if needed
        if len(res) > 500:
            idx = np.r_[:50, 50:500:5, 500:len(res):int((len(res)) / 100)]
            res = res[idx]

        np.savetxt(output_dir + '/' + fname + name, res, header=header, comments='')

    _save(train_res, '_train.txt')
    _save(val_res, '_val.txt')

    log.info('Data saved to %s/[full_]%s_[train/val].txt', output_dir, fname)
