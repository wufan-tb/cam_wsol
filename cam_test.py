import torch,torchvision,os,cv2
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
from torchvision import transforms as T
from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np
# %%
# ================定义模型结构============
class_num=2
mymodel=torchvision.models.resnet50(pretrained=True)
mymodel.fc = torch.nn.Linear(mymodel.fc.in_features, class_num)
mymodel.to('cuda')
summary(mymodel, input_size=(3,256,256))
# %%
class ClassifyDataset(Dataset):
    def __init__(self,dataset_path='./dataset/dog_vs_cat',is_train=True,resize=256,cropsize=256,
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.imgs, self.labels = self.load_dataset_folder()
        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=mean,
                                                  std=std)])

    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = Image.open(img).convert('RGB')
        img = self.transform_x(img)
        label = self.labels[idx]
        label = torch.tensor(label)
        return img.float(),label.long()

    def __len__(self):
        return len(self.imgs)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        imgs = []
        labels = []
        label_dir = os.path.join(self.dataset_path, phase)
        label_name = os.listdir(label_dir)
        index=0
        for label in label_name:
            img_dir = os.path.join(label_dir,label)
            img_fpath_list = sorted([os.path.join(img_dir, f)
                                    for f in os.listdir(img_dir)
                                    if f.endswith('.jpg')])
            imgs.extend(img_fpath_list)
            label = index
            index += 1
            labels.extend([label] * len(img_fpath_list))
        return list(imgs), list(labels)
# %%
# ==================训练模型=====================
dataset=ClassifyDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
Epoch=50
Lr=0.001
# %%
train_list=['fc.weight','fc.bias']
for name,parameters in mymodel.to('cpu').named_parameters():
    if name not in train_list:
        parameters.requires_grad = False 
params = filter(lambda p: p.requires_grad, mymodel.parameters())
optimizer = torch.optim.Adam(params, lr=Lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =Epoch)
entroy=torch.nn.CrossEntropyLoss()
mymodel.train()
mymodel.to('cuda')
for epoch in range(1,Epoch+1):
    print('||== start new epoch | {}/{} ==||'.format(epoch,Epoch))
    if epoch%(max(1,int(Epoch/10)))==0:
        torch.save(mymodel, 'checkpoint/classify_{}.pt'.format(epoch))
    for i, (img, label) in tqdm(enumerate(dataloader),'training single epoch...'): 
        pre_label=mymodel(img.to('cuda'))
        optimizer.zero_grad()
        loss=entroy(pre_label, label.to('cuda'))
        loss.backward()
        optimizer.step()
        lr = optimizer.param_groups[0]["lr"]
        if i % (max(1,int(len(dataloader)/20))) ==0:
            with open('train.log','a',encoding='utf-8') as f:
                f.writelines("epoch:{}; loss:{:.3f}; lr:{:.5f}\n".format(epoch,loss.item(),lr))
    scheduler.step()
# %% [markdown]
# # 导入训练好的模型
# %%
mymodel=torch.load('checkpoint/classify.pt')
# %%
# ==================定义t特征图和权重的获取=================
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # 如果你想feature的梯度能反向传播，那么去掉 detach（）
        activation[name] = output.detach()
    return hook
mymodel.layer4.register_forward_hook(get_activation('layer4'))

parameter_dict={}
for name,parameters in mymodel.to('cpu').named_parameters():
    parameter_dict[name]=parameters.detach().numpy()
print(parameter_dict.keys())
# %%
# ================定义推理及后处理函数====================
mymodel.eval()
mymodel.to('cuda')
def class_singleimg(img_path):
    trans=T.Compose([T.Resize(256,Image.ANTIALIAS),T.CenterCrop(256),T.ToTensor(),
                     T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
    img=Image.open(img_path).convert('RGB')
    img=trans(img)
    with torch.no_grad():
        label=mymodel(img.view(-1,3,256,256).to('cuda'))
        label=F.softmax(label,dim=-1)
        conf,pred = label.max(-1)
    return pred.item(),conf.item()

def tensor_to_img(input_tensor):
    temp=input_tensor.clone().data
    temp-=temp.min()
    temp/=temp.max()
    temp*=255
    return temp.cpu().squeeze().numpy().astype('uint8')

def merge_feature_map(all_feature,weights,pre_label):
    feature=torch.tensor(np.zeros_like(all_feature[0][0]))
    for i in range(weights.shape[1]):
        feature+=(weights[pre_label][i]*all_feature[0][i].clone().data)
    return tensor_to_img(feature)

def get_obj_box(mask,img,threshold=100):
    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        max_area=img.shape[0]*img.shape[1]
        if max_area*0.5 > cv2.contourArea(c) > max_area*0.0005:
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append(np.array([x,y,(x+w),(y+h)]))
            boxes.append(np.array([x,y,(x+w),(y+h)]))
    return boxes

def draw_box(img, boxes, label, conf):
    temp=img.copy()
    for i in range(len(boxes)):
        if conf > 0.85:
            xmin,ymin,xmax,ymax=(lambda list:[int(list[i]) for i in range(len(list))])(boxes[i])
            info = '{}:{:.2f}'.format(label,conf)
            t_size=cv2.getTextSize(info, cv2.FONT_HERSHEY_TRIPLEX, 0.5 , 1)[0]
            cv2.rectangle(temp, (xmin, ymin), (xmax, ymax), (250, 250, 0), 2)
            cv2.rectangle(temp, (xmin, ymin), (xmin + t_size[0]+3, ymin + t_size[1]+6), (250, 250, 0), -1)
            cv2.putText(temp, info, (xmin+1, ymin+t_size[1]+1), cv2.FONT_HERSHEY_TRIPLEX, 0.5, [255,255,255], 1)
    return temp
# %%
# =================测试流程=============
test_path = 'dataset/dog_vs_cat/test/'
lamel_to_name={0:'cat',1:'dog'}
test_list = os.listdir(test_path)
IMG_NUMBER = 0
RIGHT = 0
for item in test_list:
    img_path=os.path.join(test_path, item)
    img=cv2.imread(img_path)
    label,conf = class_singleimg(img_path)
    if item[0:3] == lamel_to_name[label]:
        RIGHT+=1
    IMG_NUMBER+=1
    feature_map=activation['layer4']
    cam_figure=merge_feature_map(feature_map.cpu(),parameter_dict['fc.weight'],label)
    cam_figure=cv2.resize(cam_figure,(img.shape[1],img.shape[0]),cv2.INTER_CUBIC)
    heatmap=np.uint8(cam_figure)
    heatmap=cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    
    boxes=get_obj_box(cam_figure,img,180)
    result=draw_box(img, boxes, lamel_to_name[label], conf)
    
    img_box_cam=cv2.addWeighted(result, 0.65, heatmap, 0.4, 0)
    cv2.imshow('True:{}|Pred:{}'.format(item[0:3],lamel_to_name[label]),img_box_cam)
    key=cv2.waitKey()
    if key == 113: # press 'q'
        cv2.destroyAllWindows()
        break
    cv2.destroyAllWindows()
    
print('map: {}'.format(RIGHT/(0.00001+IMG_NUMBER)))

