import collections
import tqdm
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from os.path import join
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from collections import OrderedDict as OD
from torchvision import datasets, transforms, utils

from layers import IAFLayer
from utils import *

from template_lib.utils import AverageMeter
from template_lib.trainer.base_trainer import summary_defaultdict2txtfig


# Model definition
# ----------------------------------------------------------------------------------------------
class VAE(nn.Module):
  def __init__(self, args):
    super(VAE, self).__init__()
    self.register_parameter('h', torch.nn.Parameter(torch.zeros(args.h_size)))
    self.register_parameter('dec_log_stdv', torch.nn.Parameter(torch.Tensor([0.])))

    layers = []
    # build network
    for i in range(args.depth):
      layer = []

      for j in range(args.n_blocks):
        downsample = (i > 0) and (j == 0)
        layer += [IAFLayer(args, downsample)]

      layers += [nn.ModuleList(layer)]

    self.layers = nn.ModuleList(layers)

    self.first_conv = nn.Conv2d(3, args.h_size, 4, 2, 1)
    self.last_conv = nn.ConvTranspose2d(args.h_size, 3, 4, 2, 1)

  def forward(self, input):
    # assumes input is \in [-0.5, 0.5]
    x = self.first_conv(input)
    kl, kl_obj = 0., 0.

    h = self.h.view(1, -1, 1, 1)

    for layer in self.layers:
      for sub_layer in layer:
        x = sub_layer.up(x)

    h = h.expand_as(x)
    self.hid_shape = x[0].size()

    for layer in reversed(self.layers):
      for sub_layer in reversed(layer):
        h, curr_kl, curr_kl_obj = sub_layer.down(h)
        kl += curr_kl
        kl_obj += curr_kl_obj

    x = F.elu(h)
    x = self.last_conv(x)

    x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

    return x, kl, kl_obj

  def sample(self, n_samples=64):
    h = self.h.view(1, -1, 1, 1)
    h = h.expand((n_samples, *self.hid_shape))

    for layer in reversed(self.layers):
      for sub_layer in reversed(layer):
        h, _, _ = sub_layer.down(h, sample=True)

    x = F.elu(h)
    x = self.last_conv(x)

    return x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)

  def cond_sample(self, input):
    # assumes input is \in [-0.5, 0.5]
    x = self.first_conv(input)
    kl, kl_obj = 0., 0.

    h = self.h.view(1, -1, 1, 1)

    for layer in self.layers:
      for sub_layer in layer:
        x = sub_layer.up(x)

    h = h.expand_as(x)
    self.hid_shape = x[0].size()

    outs = []

    current = 0
    for i, layer in enumerate(reversed(self.layers)):
      for j, sub_layer in enumerate(reversed(layer)):
        h, curr_kl, curr_kl_obj = sub_layer.down(h)

        h_copy = h
        again = 0
        # now, sample the rest of the way:
        for layer_ in reversed(self.layers):
          for sub_layer_ in reversed(layer_):
            if again > current:
              h_copy, _, _ = sub_layer_.down(h_copy, sample=True)

            again += 1

        x = F.elu(h_copy)
        x = self.last_conv(x)
        x = x.clamp(min=-0.5 + 1. / 512., max=0.5 - 1. / 512.)
        outs += [x]

        current += 1

    return outs


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_blocks', type=int, default=4)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--z_size', type=int, default=32)
    parser.add_argument('--h_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--free_bits', type=float, default=0.1)
    parser.add_argument('--iaf', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    return parser


# Main
# ----------------------------------------------------------------------------------------------
def main(args, myargs):
  # arguments


  # create model and ship to GPU
  model = VAE(args).cuda()
  print(model)

  # reproducibility is da best
  set_seed(0)

  opt = torch.optim.Adamax(model.parameters(), lr=args.lr)

  # create datasets / dataloaders
  scale_inv = lambda x: x + 0.5
  ds_transforms = transforms.Compose([transforms.ToTensor(), lambda x: x - 0.5])
  kwargs = {'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': True}

  train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.dataset_dir, train=True,
                                                              download=True, transform=ds_transforms),
                                             batch_size=args.batch_size, shuffle=True, **kwargs)

  test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.dataset_dir, train=False,
                                                             download=True, transform=ds_transforms),
                                            batch_size=args.batch_size, shuffle=False, **{**kwargs, 'drop_last': False})

  # spawn writer
  model_name = 'NB{}_D{}_Z{}_H{}_BS{}_FB{}_LR{}_IAF{}'.format(args.n_blocks, args.depth, args.z_size, args.h_size,
                                                              args.batch_size, args.free_bits, args.lr, args.iaf)

  model_name = 'test' if args.debug else model_name
  # log_dir = join('runs', model_name)
  # sample_dir = join(log_dir, 'samples')
  # writer = SummaryWriter(log_dir=log_dir)
  log_dir = f'{myargs.args.outdir}/iaf'
  os.makedirs(log_dir, exist_ok=True)
  sample_dir = myargs.args.imgdir
  writer = myargs.writer
  maybe_create_dir(sample_dir)

  print_and_save_args(args, log_dir)
  print('logging into %s' % log_dir)
  maybe_create_dir(sample_dir)
  best_test = float('inf')

  print('starting training')
  kl_meter = AverageMeter()
  bpd_meter = AverageMeter()
  elbo_meter = AverageMeter()
  kl_obj_meter = AverageMeter()
  log_pxz_meter = AverageMeter()

  for epoch in range(args.n_epochs):
    print(f'\nEpoch [{epoch}/{args.n_epochs}]')
    model.train()
    train_log = reset_log()
    kl_meter.reset()
    bpd_meter.reset()
    elbo_meter.reset()
    kl_obj_meter.reset()
    log_pxz_meter.reset()
    summary_d = collections.defaultdict(dict)

    for batch_idx, (input, _) in enumerate(tqdm.tqdm(train_loader, file=myargs.stdout,
                                                     desc=myargs.args.time_str_suffix)):
      if args.train_dummy and batch_idx != 0:
        break
      input = input.cuda()
      x, kl, kl_obj = model(input)

      log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
      loss = (kl_obj - log_pxz).sum() / x.size(0)
      elbo = (kl - log_pxz)
      bpd = elbo / (32 * 32 * 3 * np.log(2.))

      opt.zero_grad()
      loss.backward()
      opt.step()

      # train_log['kl'] += [kl.mean()]
      # train_log['bpd'] += [bpd.mean()]
      # train_log['elbo'] += [elbo.mean()]
      # train_log['kl obj'] += [kl_obj.mean()]
      # train_log['log p(x|z)'] += [log_pxz.mean()]
      # for key, value in train_log.items():
      #   print_and_log_scalar(writer, 'train/%s' % key, value, epoch)
      # print()

      kl_meter.update(kl.mean().item())
      bpd_meter.update(bpd.mean().item())
      elbo_meter.update(elbo.mean().item())
      kl_obj_meter.update(kl_obj.mean().item())
      log_pxz_meter.update(log_pxz.mean().item())

    summary_d['kl_meter']['train'] = kl_meter.avg
    summary_d['bpd_meter']['train'] = bpd_meter.avg
    summary_d['elbo_meter']['train'] = elbo_meter.avg
    summary_d['kl_obj_meter']['train'] = kl_obj_meter.avg
    summary_d['log_pxz_meter']['train'] = log_pxz_meter.avg

    model.eval()
    test_log = reset_log()
    kl_meter.reset()
    bpd_meter.reset()
    elbo_meter.reset()
    kl_obj_meter.reset()
    log_pxz_meter.reset()

    with torch.no_grad():
      for batch_idx, (input, _) in enumerate(tqdm.tqdm(test_loader, file=myargs.stdout)):
        input = input.cuda()
        x, kl, kl_obj = model(input)

        log_pxz = logistic_ll(x, model.dec_log_stdv, sample=input)
        loss = (kl_obj - log_pxz).sum() / x.size(0)
        elbo = (kl - log_pxz)
        bpd = elbo / (32 * 32 * 3 * np.log(2.))

        # test_log['kl'] += [kl.mean()]
        # test_log['bpd'] += [bpd.mean()]
        # test_log['elbo'] += [elbo.mean()]
        # test_log['kl obj'] += [kl_obj.mean()]
        # test_log['log p(x|z)'] += [log_pxz.mean()]

        kl_meter.update(kl.mean().item())
        bpd_meter.update(bpd.mean().item())
        elbo_meter.update(elbo.mean().item())
        kl_obj_meter.update(kl_obj.mean().item())
        log_pxz_meter.update(log_pxz.mean().item())

      summary_d['kl_meter']['test'] = kl_meter.avg
      summary_d['bpd_meter']['test'] = bpd_meter.avg
      summary_d['elbo_meter']['test'] = elbo_meter.avg
      summary_d['kl_obj_meter']['test'] = kl_obj_meter.avg
      summary_d['log_pxz_meter']['test'] = log_pxz_meter.avg

      summary_defaultdict2txtfig(summary_d, prefix='', step=epoch,
                                 textlogger=myargs.textlogger, save_fig_sec=60)

      all_samples = model.cond_sample(input)
      # save reconstructions
      out = torch.stack((x, input))  # 2, bs, 3, 32, 32
      out = out.transpose(1, 0).contiguous()  # bs, 2, 3, 32, 32
      out = out.view(-1, x.size(-3), x.size(-2), x.size(-1))

      all_samples += [x]
      all_samples = torch.stack(all_samples)  # L, bs, 3, 32, 32
      all_samples = all_samples.transpose(1, 0)
      all_samples = all_samples.contiguous()  # bs, L, 3, 32, 32
      all_samples = all_samples.view(-1, x.size(-3), x.size(-2), x.size(-1))

      save_image(scale_inv(all_samples), join(sample_dir, 'test_levels_{}.png'.format(epoch)), nrow=12)
      save_image(scale_inv(out), join(sample_dir, 'test_recon_{}.png'.format(epoch)), nrow=12)
      save_image(scale_inv(model.sample(64)), join(sample_dir, 'sample_{}.png'.format(epoch)), nrow=8)

    # for key, value in test_log.items():
    #   print_and_log_scalar(writer, 'test/%s' % key, value, epoch)
    # print()

    # current_test = sum(test_log['bpd']) / batch_idx
    current_test = bpd_meter.avg

    if current_test < best_test:
      best_test = current_test
      print('saving best model')
      torch.save(model.state_dict(), join(log_dir, 'best_model.pth'))


def run(argv_str=None, return_args=False):
  from template_lib.utils.config import parse_args_and_setup_myargs, config2args
  from template_lib.utils.modelarts_utils import prepare_dataset
  run_script = os.path.relpath(__file__, os.getcwd())
  args1, myargs, _ = parse_args_and_setup_myargs(argv_str, run_script=run_script, start_tb=False)
  myargs.args = args1
  myargs.config = getattr(myargs.config, args1.command)

  if return_args:
    return args1, myargs

  parser = build_parser()
  args = parser.parse_args([])
  args = config2args(myargs.config, args)

  main(args, myargs)
  pass


if __name__ == '__main__':
  run()
