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
    parser = argparse.ArgumentParser(description='Pytorch 12 tasks Baseline Training')

    # save and load
    parser.add_argument('--data-dir', dest='data_dir', type=str)
    parser.set_defaults(data_dir='decathlon-1.0-data')

    parser.add_argument('--model-weight-path', dest='model_weight_path', type=str)
    parser.set_defaults(model_weight_path=None)

    parser.add_argument('--log-dir', dest='log_dir', type=str)
    parser.set_defaults(log_dir='log_save')

    parser.add_argument('--model_save_dir', dest='model_save_dir', type=str)
    parser.set_defaults(model_save_dir='model_weights')

    # network
    parser.add_argument('--depth', dest='depth', type=int)
    parser.set_defaults(depth=28)

    parser.add_argument('--widen-factor', dest='widen_factor', type=int)
    parser.set_defaults(widen_factor=1)

    # training
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
    epoch_step = json.loads(args.lr_step)
    prj_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    log_dir = os.path.join(prj_dir, args.log_dir)
    model_weight_path = os.path.join(prj_dir, args.model_weight_path)
    model_save_dir = os.path.join(prj_dir, args.model_save_dir)
    data_dir = os.path.join(prj_dir, args.data_dir)

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    data_name = ['cifar100', 'aircraft', 'daimlerpedcls', 'dtd', 'gtsrb', 'omniglot',
                 'svhn', 'ucf101', 'vgg-flowers', 'cifar10', 'caltech256', 'sketches']

    data_class = [100, 100, 2, 47, 43, 1623, 10, 101, 102, 10, 257, 250]
    im_train_set = [0] * len(data_class)
    im_test_set = [0] * len(data_class)
    total_epoch = args.epoch
    avg_cost = np.zeros([total_epoch, len(data_class), 4], dtype=np.float32)
    for i in range(len(data_class)):
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

        # define WideResNet model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        WideResNet_ins = WideResNet(depth=args.depth, widen_factor=args.widen_factor, num_classes=data_class[i]).to(device)
        optimizer = optim.SGD(WideResNet_ins.parameters(), lr=args.lr, weight_decay=args.weight_decay, nesterov=True, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, epoch_step, gamma=args.lr_drop_ratio)

        if model_weight_path is not None:
            WideResNet_ins.load_state_dict(torch.load(model_weight_path))
            print('[*]Model loaded : ({})'.format(os.path.basename(model_weight_path)))

        print('Training DATASET:{}'.format(data_name[i]))
        time_onedataset = time.time()
        for index in range(total_epoch):
            scheduler.step()
            time_epoch = time.time()

            cost = np.zeros(2, dtype=np.float32)
            train_dataset = iter(im_train_set[i])
            train_batch = len(train_dataset)
            WideResNet_ins.train()
            for k in range(train_batch):
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
                avg_cost[index, i, :2] += cost / train_batch

            # evaluating test data
            WideResNet_ins.eval()
            test_dataset = iter(im_test_set[i])
            test_batch = len(test_dataset)
            with torch.no_grad():
                for k in range(test_batch):
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
                    avg_cost[index, i, 2:] += cost / test_batch

            print('EPOCH: {:04d} | DATASET: {:s} || TRAIN: {:.4f} {:.4f} || TEST: {:.4f} {:.4f} TIME: {:.2f} minutes {:.2f} seconds'
                  .format(index, data_name[i], avg_cost[index, i, 0], avg_cost[index, i, 1],
                          avg_cost[index, i, 2], avg_cost[index, i, 3], (time.time()-time_epoch)//60, (time.time()-time_epoch)%60))
            print('='*100)

            if not os.path.exists(os.path.join(model_save_dir, 'baseline', data_name[i])):
                os.mkdir(os.path.join(model_save_dir, 'baseline', data_name[i]))

            if index % 5 == 0:
                torch.save(WideResNet_ins.state_dict(),
                           os.path.join(model_save_dir, 'baseline', data_name[i],
                                        'wideresnet{}_{}_{}.pt'.format(args.depth, args.widen_factor, index)))

        torch.save(WideResNet_ins.state_dict(),
                   os.path.join(model_save_dir, 'baseline', data_name[i],
                                'wideresnet{}_{}_final.pt'.format(args.depth, args.widen_factor)))
        print('DATASET: {:s} : Time consumed: {:.2f} hours {:.2f} minutes {:.2f} seconds'
              .format(data_name[i], (time.time()-time_onedataset)//3600, ((time.time()-time_onedataset)%3600)//60, (time.time()-time_onedataset)%60))

    np.save(os.path.join(log_dir, 'wideresnet{}_{}_train_log_baseline.npy'.format(args.depth, args.widen_factor)))


if __name__ == '__main__':
    main()

