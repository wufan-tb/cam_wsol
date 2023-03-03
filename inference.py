import argparse
import os
import subprocess

import cv2
import torch
from tqdm import tqdm

from utils import *


def inference(args):

    print('== loading model... ==')
    mymodel = torch.load(args.ckpt)

    activation = {}

    def get_activation(name):

        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    mymodel.layer4.register_forward_hook(get_activation('layer4'))

    parameter_dict = {}
    for name, parameters in mymodel.to('cpu').named_parameters():
        parameter_dict[name] = parameters.detach().numpy()

    sys_cmd = 'rm -rf {}'.format(args.output)
    child = subprocess.Popen(sys_cmd, shell=True)
    child.wait()
    os.makedirs(args.output, exist_ok=True)

    if os.path.isdir(args.input):
        test_list = [
            os.path.join(args.input, item) for item in os.listdir(args.input)
        ]
    else:
        test_list = [args.input]

    with torch.no_grad():
        mymodel.eval()
        mymodel.to('cuda')
        for i, img_path in tqdm(enumerate(test_list),
                                desc='processing ...',
                                total=len(test_list)):

            img = cv2.imread(img_path)
            label, conf = classify_singleimg(img_path, mymodel, args.device)

            feature_map = activation['layer4']
            cam_figure = merge_feature_map(feature_map.cpu(),
                                           parameter_dict['fc.weight'], label)
            cam_figure = cv2.resize(cam_figure, (img.shape[1], img.shape[0]),
                                    cv2.INTER_CUBIC)
            heatmap = np.uint8(cam_figure)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            boxes = get_obj_box(cam_figure, img, 180)
            result = draw_box(img, boxes, mymodel.class_names[label], conf)

            img_box_cam = cv2.addWeighted(result, 0.65, heatmap, 0.4, 0)
            cv2.imwrite(os.path.join(args.output, f'result_{i}.jpg'),
                        img_box_cam)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt',
                        type=str,
                        default='checkpoint/dog_vs_cat_5.pt',
                        help='dataset path')

    parser.add_argument('--input',
                        type=str,
                        default='data/dog_vs_cat/test',
                        help='path to save ckpt files')
    parser.add_argument('--output',
                        type=str,
                        default='demo',
                        help='path to save ckpt files')

    parser.add_argument('--device',
                        default='cuda',
                        help='cuda device, i.e. cuda or cpu')

    args = parser.parse_args()
    assert args.input != args.output, 'input dir should not be equal to output dir, please check again.'
    print('== init training args ==')
    print(args)

    inference(args)
