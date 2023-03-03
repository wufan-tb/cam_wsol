import argparse
import os

import cv2
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import dog_vs_cat_Dataset


def train(args):

    print('== loading model... ==')
    mymodel = torchvision.models.resnet50(pretrained=True)

    train_dataset = dog_vs_cat_Dataset(args.data_path)
    dataloader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)

    mymodel.fc = torch.nn.Linear(mymodel.fc.in_features,
                                 len(train_dataset.class_names))
    mymodel.class_names = train_dataset.class_names

    train_list = ['fc.weight', 'fc.bias']
    for name, parameters in mymodel.named_parameters():
        if name not in train_list:
            parameters.requires_grad = False
    params = filter(lambda p: p.requires_grad, mymodel.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epoch)

    entroy = torch.nn.CrossEntropyLoss()

    mymodel.train()
    mymodel.to('cuda')

    for epoch in range(1, args.epoch + 1):
        print('\n== start new epoch | {}/{} =='.format(epoch, args.epoch))
        total = 0
        correct = 0
        for i, (img, label) in enumerate(dataloader):
            pre_label = mymodel(img.to('cuda'))
            optimizer.zero_grad()
            loss = entroy(pre_label, torch.squeeze(label).to('cuda'))
            loss.backward()
            optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            if i % (max(1, int(len(dataloader) / 5))) == 0:
                if args.to_log:
                    with open('train.log', 'a', encoding='utf-8') as f:
                        f.writelines(
                            'epoch:{}; iter:{} loss:{:.3f}; lr:{:.5f}\n'.
                            format(epoch, i, loss.item(), lr))
                else:
                    print(
                        '    epoch:{}; iter:{}; loss:{:.3f}; lr:{:.5f}'.format(
                            epoch, i, loss.item(), lr))

            total += label.size(0)
            _, predicted = torch.max(pre_label.data, 1)
            predicted = 1 * (predicted > 0.5)
            correct += (torch.squeeze(
                predicted.cpu()) == torch.squeeze(label)).sum().item()

        print(
            f'== epoch: {epoch}, training acc: {round(100 * correct / total, 2)} =='
        )
        scheduler.step()

        if epoch % (max(1, int(args.epoch / args.max_ckpt_nums))) == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(mymodel,
                       os.path.join(args.save_dir, f'dog_vs_cat_{epoch}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='data/dog_vs_cat',
                        help='dataset path')

    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--epoch', type=int, default=5, help='training epoch')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='batch size')

    parser.add_argument('--save_dir',
                        type=str,
                        default='checkpoint',
                        help='path to save ckpt files')
    parser.add_argument('--max_ckpt_nums',
                        type=int,
                        default=5,
                        help='max ckpt nums to save')

    parser.add_argument('--device',
                        default='cuda',
                        help='cuda device, i.e. cuda or cpu')

    parser.add_argument('--to_log',
                        action='store_true',
                        help='save result to video')

    args = parser.parse_args()
    print('== init training args ==')
    print(args)

    train(args)
