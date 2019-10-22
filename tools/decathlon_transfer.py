# -*- coding: utf-8 -*-
import torch
import torchvision
import numpy as np
import torch.optim as optim
import os
import time
import argparse
import json

from wideresnet import *
from utils import data_transform


# main()
def main():
    parser = argparse.ArgumentParser(description='Pytorch wideresnet finetuning')

    # save and load
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.set_defaults(data_dir='decathlon-1.0-data')

    parser.add_argument('--model-weight-dir', dest='model_weight_dir', type=str)
    parser.set_defaults(model_weight_path='model_weights')

    parser.add_argument('--transfer-result-dir', dest='transfer_result_dir', type=str)
    parser.set_defaults(transfer_result_dir='transfer_result_fc_all')

    # network
    parser.add_argument('--depth', dest='depth', type=int)
    parser.set_defaults(depth=28)

    parser.add_argument('--widen-factor', dest='widen_factor', type=int)
    parser.set_defaults(widen_factor=1)

    # finetuning
    parser.add_argument('--fc', action='store_true')
    parser.set_defaults(target=False)

    parser.add_argument('--lr', dest='lr', type=float)
    parser.set_defaults(lr=0.1)

    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.set_defaults(batch_size=128)

    parser.add_argument('--epoch', dest='epoch', type=int)
    parser.set_defaults(epoch=200)

    parser.add_argument('--weight-decay', dest='weight_decay', type=float)
    parser.set_defaults(weight_decay=5e-4)

    parser.add_argument('--lr-step', dest='lr_step', type=str)
    parser.set_defaults(lr_step='[60, 120, 160]')

    parser.add_argument('--lr-drop-ratio', dest='lr_drop_ratio', type=float)
    parser.set_defaults(lr_drop_ratio=0.2)

    parser.add_argument('--gpu', dest='gpu', type=str)
    parser.set_defaults(gpu='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    model_weight_dir = os.path.join(prj_dir, args.model_weight_dir)
    data_dir = os.path.join(prj_dir, args.data_dir)
    transfer_result_dir = os.path.join(prj_dir, args.transfer_result_dir)

    if not os.path.exists(transfer_result_dir):
        os.mkdir(transfer_result_dir)
    epoch_step = json.loads(args.lr_step)
    data_name = ['cifar100', 'aircraft', 'daimlerpedcls', 'dtd',
                 'gtsrb', 'omniglot', 'svhn', 'ucf101', 'vgg-flowers',
                 'cifar10', 'caltech256', 'sketches']
    data_class = [100, 100, 2, 47, 43, 1623, 10, 101, 102, 10, 257, 250]
    im_train_set = [0] * len(data_class)
    im_test_set = [0] * len(data_class)
    total_epoch = args.epoch
    avg_cost = np.zeros([total_epoch, len(data_class), len(data_class), 4], dtype=np.float32) # :[epoch, target, source, 4]
    if args.fc:
        trainlog_save = 'wideresnet{}_{}_cost_fc.npy'.format(args.depth, args.widen_factor)
    else:
        trainlog_save = 'wideresnet{}_{}_cost_all.npy'.format(args.depth, args.widen_factor)
    print('Fintuned using {}'.format(data_dir))
    for i in range(len(data_class)): # 12 tasks
        im_train_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, data_name[i], 'train'),
                                                      transform=data_transform(data_dir, data_name[i], train=True)),
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=4, pin_memory=True)
        im_test_set[i] = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(os.path.join(data_dir, data_name[i], 'val'),
                                                     transform=data_transform(data_dir, data_name[i], train=False)),
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=4, pin_memory=True)

        # define WRN model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WideResNet_ins = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=data_class[i]).to(device)

        para_optim = []
        count = 0
        for k in WideResNet_ins.parameters():
            count += 1
            if count > 106:
                para_optim.append(k)
            else:
                k.requires_grad = False

        # Fine-tune fully-connected ?
        if args.fc:
            optimizer = optim.SGD(para_optim, lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        else:
            optimizer = optim.SGD(WideResNet_ins.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_step, gamma=args.lr_drop_ratio)

        if not os.path.exists(os.path.join(transfer_result_dir, data_name[i])):
            os.mkdir(os.path.join(transfer_result_dir, data_name[i]))

        for k, src in enumerate(data_name):
            if src == data_name[i]:
                continue

            # load src's model params
            if model_weight_dir is not None:
                if args.fc:
                    pretrained_dict = torch.load(os.path.join(model_weight_dir, 'baseline', src,
                                                              'wideresnet{}_{}_final.pt'.format(args.depth, args.widen_factor)))
                    model_dict = WideResNet_ins.state_dict()
                    pretrained_dict = {k_: v_ for k_, v_ in pretrained_dict.items() if k_ != 'linear.0.weight' and k_ != 'linear.0.bias'}
                    model_dict.update(pretrained_dict)
                    WideResNet_ins.load_state_dict(model_dict)
                else:
                    WideResNet_ins.load_state_dict(torch.load(os.path.join(model_weight_dir, 'baseline', src,
                                                              'wideresnet{}_{}_final.pt'.format(args.depth, args.widen_factor))))
                print('[*] Loaded {} ==> {}'.format(src, data_name[i]))
            else:
                raise IOError('Transfer must have source!')

            print('Start Finetuning DATASET:{} from Pretrained-MODEL:{}'.format(data_name[i], src))
            time_onedataset = time.time()
            for index in range(total_epoch):
                scheduler.step()
                time_epoch = time.time()

                cost = np.zeros(2, dtype=np.float32)
                train_dataset = iter(im_train_set[i])
                train_batch = len(train_dataset)
                WideResNet_ins.train()
                for _ in range(train_batch):
                    train_data, train_label = train_dataset.next()
                    train_label = train_label.type(torch.LongTensor)
                    train_data, train_label = train_data.to(device), train_label.to(device)
                    train_pred1 = WideResNet_ins(train_data)[0]

                    # reset optimizer with zero gradient
                    optimizer.zero_grad()
                    train_loss1 = WideResNet_ins.model_fit(train_pred1, train_label, device=device, num_output=data_class[i])
                    train_loss = torch.mean(train_loss1)
                    train_loss.backward()
                    optimizer.step()

                    # calculate training loss and accuracy
                    train_predict_label1 = train_pred1.data.max(1)[1]
                    train_acc1 = train_predict_label1.eq(train_label).sum().item() / train_data.shape[0]

                    cost[0] = torch.mean(train_loss1).item()
                    cost[1] = train_acc1
                    avg_cost[index, i, k, 0:2] += cost / train_batch

                # evaluating test data
                WideResNet_ins.eval()
                test_dataset = iter(im_test_set[i])
                test_batch = len(test_dataset)
                for _ in range(test_batch):
                    test_data, test_label = test_dataset.next()
                    test_label = test_label.type(torch.LongTensor)
                    test_data, test_label = test_data.to(device), test_label.to(device)
                    test_pred1 = WideResNet_ins(test_data)[0]

                    test_loss1 = WideResNet_ins.model_fit(test_pred1, test_label, device=device, num_output=data_class[i])

                    # calculate testing loss and accuracy
                    test_predict_label1 = test_pred1.data.max(1)[1]
                    test_acc1 = test_predict_label1.eq(test_label).sum().item() / test_data.shape[0]

                    cost[0] = torch.mean(test_loss1).item()
                    cost[1] = test_acc1
                    avg_cost[index, i, k, 2:] += cost / test_batch

                print('EPOCH: {:04d} | DATASET: {:s} Finetuned from {:s} || TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} TIME: {:.2f} minutes {:.2f} seconds'
                      .format(index, data_name[i], src, avg_cost[index, i, k, 0], avg_cost[index, i, k, 1],
                              avg_cost[index, i, k, 2], avg_cost[index, i, k, 3], (time.time()-time_epoch)//60, (time.time()-time_epoch)%60))
                print('='*100)

                if not os.path.exists(os.path.join(transfer_result_dir, data_name[i], src)):
                    os.mkdir(os.path.join(transfer_result_dir, data_name[i], src))

                if index % 5 == 0:
                    torch.save(WideResNet_ins.state_dict(), os.path.join(transfer_result_dir, data_name[i], src, 'wideresnet{}_{}_{}.pt'.format(args.depth, args.widen_factor, index)))

            torch.save(WideResNet_ins.state_dict(), os.path.join(transfer_result_dir, data_name[i], src, 'wideresnet{}_{}_final.pt'.format(args.depth, args.widen_factor)))
            print('DATASET: {:s} Finetuned from {:s} : Time consumed: {:.2f} minutes {:.2f} seconds'.format(data_name[i], src, (time.time()-time_onedataset)//60, (time.time()-time_onedataset)%60))
            cost_ = avg_cost[:, i, k, :]
            np.save(os.path.join(transfer_result_dir, data_name[i], src, 'cost.npy'), cost_)

    np.save(os.path.join(transfer_result_dir, trainlog_save), avg_cost)


if __name__ == '__main__':
    main()
