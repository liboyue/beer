import os

import torch
import torch.nn as nn

from beer import log
import beer.experiment_utils as utils

torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


parser = utils.get_parser()
args = utils.parse_args(parser)
if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""
else:
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        os.environ['CUDA_VISIBLE_DEVICES'] = devices[args.rank % len(devices)]
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.rank % 4)

args.output_dir = f'experiments/{args.optimizer}_{args.world_size}_agents_{args.graph_type}_graph_lr_{args.lr:.6f}_gamma_{args.gamma:.5f}_{args.compression_type}_{args.compression_params[0]:d}'
if not args.cpu:
    args.output_dir += '_cuda'

utils.init(args)

train_loader, val_loader = utils.load_mnist(args.rank, args.world_size, args.batch_size, sort=True)
train_loader_for_val, _ = utils.load_mnist(args.rank, args.world_size, 300)

log.info(train_loader.dataset.targets.unique())
log.info(train_loader.dataset.targets.shape[0])

model = Net().to(args.device)
if args.optimizer == 'BEER':
    args.use_ref_module = True
args.criterion = nn.CrossEntropyLoss()

model, optimizer = utils.wrap_model(model, args)
log.info('Model is on %s, size %d', next(model.parameters()).device, model.flat_parameters.shape[0])

classes = [int(i) for i in range(10)]
criterion = nn.CrossEntropyLoss()

if args.optimizer == 'BEER':
    from beer.utils import flatten_tensors
    model.train()
    model.zero_grad()
    for i, (data, target) in enumerate(train_loader):
        target = target.to(device=args.device, non_blocking=True)
        data = data.to(device=args.device, non_blocking=True)
        output = model.module(data)
        loss = criterion(output, target)
        loss.backward()

    optimizer.V = flatten_tensors([t.grad for t in model.module.parameters()]).to(args.device) / (i + 1)
    model.zero_grad()
    optimizer.init()

train_res, val_res = utils.train(model, criterion, optimizer, train_loader, args, train_loader_for_val=train_loader_for_val,
                                 exp_name=f'mnist_{args.optimizer}', val_loader=val_loader, classes=classes)

log.info('Process %d exited', args.rank)
