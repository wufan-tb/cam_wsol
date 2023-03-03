import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm


def classify_singleimg(img_path, mymodel, device):
    trans = T.Compose([
        T.Resize(256, Image.ANTIALIAS),
        T.CenterCrop(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = trans(img)

    label = mymodel(img.view(-1, 3, 256, 256).to(device))
    label = F.softmax(label, dim=-1)
    conf, pred = label.max(-1)
    return pred.item(), conf.item()


def tensor_to_img(input_tensor):
    temp = input_tensor.clone().data
    temp -= temp.min()
    temp /= temp.max()
    temp *= 255
    return temp.cpu().squeeze().numpy().astype('uint8')


def merge_feature_map(all_feature, weights, pre_label):
    feature = torch.tensor(np.zeros_like(all_feature[0][0]))
    for i in range(weights.shape[1]):
        feature += (weights[pre_label][i] * all_feature[0][i].clone().data)
    return tensor_to_img(feature)


def get_obj_box(mask, img, threshold=150):
    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        max_area = img.shape[0] * img.shape[1]
        if max_area * 0.5 > cv2.contourArea(c) > max_area * 0.0005:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append(np.array([x, y, (x + w), (y + h)]))
            boxes.append(np.array([x, y, (x + w), (y + h)]))
    return boxes


def draw_box(img, boxes, label, conf):
    temp = img.copy()
    for i in range(len(boxes)):
        if conf > 0.85:
            xmin, ymin, xmax, ymax = (
                lambda list: [int(list[i])
                              for i in range(len(list))])(boxes[i])
            info = '{}:{:.2f}'.format(label, conf)
            t_size = cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.5, 1)[0]
            cv2.rectangle(temp, (xmin, ymin), (xmax, ymax), (250, 250, 0), 2)
            cv2.rectangle(temp, (xmin, ymin),
                          (xmin + t_size[0] + 3, ymin + t_size[1] + 6),
                          (250, 250, 0), -1)
            cv2.putText(temp, info, (xmin + 1, ymin + t_size[1] + 1),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)
    return temp
