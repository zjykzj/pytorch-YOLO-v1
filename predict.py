# encoding:utf-8
#
# created by xiongzihua
#

import os
import torch
from torch.autograd import Variable
import torch.nn as nn

from net import vgg16, vgg16_bn
from resnet_yolo import resnet50
import torchvision.transforms as transforms
import cv2
import numpy as np

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

Color = [[0, 0, 0],
         [128, 0, 0],
         [0, 128, 0],
         [128, 128, 0],
         [0, 0, 128],
         [128, 0, 128],
         [0, 128, 128],
         [128, 128, 128],
         [64, 0, 0],
         [192, 0, 0],
         [64, 128, 0],
         [192, 128, 0],
         [64, 0, 128],
         [192, 0, 128],
         [64, 128, 128],
         [192, 128, 128],
         [0, 64, 0],
         [128, 64, 0],
         [0, 192, 0],
         [128, 192, 0],
         [0, 64, 128]]


def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    boxes = []
    cls_indexs = []
    probs = []

    grid_num = 14
    cell_size = 1. / grid_num

    pred = pred.data
    pred = pred.squeeze(0)  # 7x7x30

    contain1 = pred[:, :, 4].unsqueeze(2)
    contain2 = pred[:, :, 9].unsqueeze(2)
    contain = torch.cat((contain1, contain2), 2)
    mask1 = contain > 0.1  # 大于阈值
    mask2 = (contain == contain.max())  # we always select the best contain_prob what ever it>0.9
    mask = (mask1 + mask2).gt(0)

    conf_thre = 0.2

    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i, j, b] == 1:
                    # print(i,j,b)
                    box = pred[i, j, b * 5:b * 5 + 4]
                    contain_prob = torch.FloatTensor([pred[i, j, b * 5 + 4]])
                    xy = torch.FloatTensor([j, i]) * cell_size  # cell左上角  up left of cell
                    box[:2] = box[:2] * cell_size + xy  # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())  # 转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5 * box[2:]
                    box_xy[2:] = box[:2] + 0.5 * box[2:]
                    max_prob, cls_index = torch.max(pred[i, j, 10:], 0)
                    # if float((contain_prob * max_prob)[0]) > 0.1:
                    # if float(contain_prob[0] * max_prob) > 0.1:
                    if float(contain_prob[0] * max_prob) > conf_thre:
                        # boxes.append(box_xy.view(1, 4))
                        boxes.append(box_xy)
                        cls_indexs.append(cls_index)
                        probs.append(contain_prob[0] * max_prob)
    if len(boxes) == 0:
        boxes = torch.zeros((1, 4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.stack(boxes)  # (n,4)
        probs = torch.stack(probs)  # (n,)
        cls_indexs = torch.stack(cls_indexs)  # (n,)
        # boxes = torch.cat(boxes, 0)  # (n,4)
        # probs = torch.cat(probs, 0)  # (n,)
        # cls_indexs = torch.cat(cls_indexs, 0)  # (n,)
    keep = nms(boxes, probs, threshold=0.5)
    return boxes[keep], cls_indexs[keep], probs[keep]


def bb_intersection_over_union(boxA, boxB):
    # boxA = [int(x) for x in boxA]
    # boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    boxAArea = max(0, boxAArea)
    boxBArea = max(0, boxBArea)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def nms(bboxes, scores, threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(0, descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break

        # xx1 = x1[order[1:]].clamp(min=x1[i])
        # yy1 = y1[order[1:]].clamp(min=y1[i])
        # xx2 = x2[order[1:]].clamp(max=x2[i])
        # yy2 = y2[order[1:]].clamp(max=y2[i])
        #
        # w = (xx2 - xx1).clamp(min=0)
        # h = (yy2 - yy1).clamp(min=0)
        # inter = w * h
        #
        # ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ovr = list()
        for idx in range(1, len(order)):
            ovr.append(bb_intersection_over_union(bboxes[i], bboxes[order[idx]]))
        ovr = torch.from_numpy(np.array(ovr))

        ids = (ovr <= threshold)
        ids = ids.nonzero()
        ids = ids.squeeze(-1)
        if ids.numel() == 0:
            # 说明置信度低的预测框和置信度高的预测框都不重叠
            break
        # ids是从0开始的，所以下标加1
        order = order[ids + 1]
    return torch.LongTensor(keep)


#
# start predict one image
#
def predict_gpu(model, image_name, device, root_path='./'):
    result = []
    # image = cv2.imread(root_path + image_name)
    img_path = os.path.join(root_path, image_name)
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    img = cv2.resize(image, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean = (123, 117, 104)  # RGB
    img = img - np.array(mean, dtype=np.float32)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(img)
    with torch.no_grad():
        img = img[None, :, :, :]
    # img = Variable(img[None, :, :, :], volatile=True)
    # img = img.cuda()

    pred = model(img.to(device))  # 1x7x7x30
    pred = pred.cpu()
    boxes, cls_indexs, probs = decoder(pred)

    for i, box in enumerate(boxes):
        x1 = int(box[0] * w)
        x2 = int(box[2] * w)
        y1 = int(box[1] * h)
        y2 = int(box[3] * h)
        cls_index = cls_indexs[i]
        cls_index = int(cls_index)  # convert LongTensor to int
        prob = probs[i]
        prob = float(prob)
        result.append([(x1, y1), (x2, y2), VOC_CLASSES[cls_index], image_name, prob])
    return result


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = resnet50()
    print('load model...')
    model.load_state_dict(torch.load('best.pth'))
    model.eval()

    model = model.to(device)
    # model.cuda()

    # image_name = 'dog.jpg'
    image_name = 'person.jpg'
    image = cv2.imread(image_name)

    print('predicting...')
    result = predict_gpu(model, image_name, device)

    print("drawing...")
    for left_up, right_bottom, class_name, _, prob in result:
        color = Color[VOC_CLASSES.index(class_name)]
        cv2.rectangle(image, left_up, right_bottom, color, 2)
        label = class_name + str(round(prob, 2))
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        p1 = (left_up[0], left_up[1] - text_size[1])
        cv2.rectangle(image, (p1[0] - 2 // 2, p1[1] - 2 - baseline), (p1[0] + text_size[0], p1[1] + text_size[1]),
                      color, -1)
        cv2.putText(image, label, (p1[0], p1[1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, 8)
    cv2.imwrite('result.jpg', image)
