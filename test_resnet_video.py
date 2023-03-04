import torch
import torch.nn as nn
from torchvision import transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import numpy as np
from my_residual_models import MyResnet


class AlexnetInference:
    def __init__(self, model_path):
        state_dict = torch.load(model_path)
        self.alexNet = MyResnet(3)
        self.alexNet.load_state_dict(state_dict)
        self.transform = transforms.Compose([
            transforms.Resize(size=(608, 608), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.48670617, 0.47717795, 0.47324306], [0.2036544, 0.20564124, 0.2122561]),
        ])
        self.alexNet.eval()
        self.alexNet.cuda()

    def draw_result(self, frame_read, label):
        img_h = frame_read.shape[0]
        img_w = frame_read.shape[1]
        pt1 = (int(img_w / 4), int(img_h / 4))
        pt2 = (int(3 * img_w / 4), int(3 * img_h / 4))
        if label == 0:
            cv2.rectangle(frame_read, pt1, pt2, (255, 0, 0), 6)
            cv2.putText(frame_read, " [NONE]",
                        (int(img_w / 4), int(img_h / 3 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 3, [255, 0, 0], 3)
        elif label == 1:
            cv2.rectangle(frame_read, pt1, pt2, (0, 0, 255), 6)
            cv2.putText(frame_read, " [With_Container]",
                        (int(img_w / 4), int(img_h / 3 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 0, 255], 3)
        else:
            cv2.rectangle(frame_read, pt1, pt2, (0, 255, 0), 6)
            cv2.putText(frame_read, " [Without_Container]",
                        (int(img_w / 4), int(img_h / 3 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 3, [0, 255, 0], 3)
        return frame_read

    def inference_img(self, frame_read):
        image = Image.fromarray(cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB))
        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0)
        #inputdata = torch.autograd.Variable(tensor, requires_grad=False)
        inputdata = tensor.cuda()
        outputdata = self.alexNet(inputdata)[0]
        ps = torch.exp(outputdata)
        top_p, top_class = ps.topk(1, dim=1)
        top_class_numpy = top_class.cpu().numpy()
        label = top_class_numpy[0][0]
        print(top_p)
        print(top_class_numpy[0][0])
        return label


# load pretrained
if __name__ == '__main__':
    filePath = r"E:\zhonghuan_door\202207012006.mp4"
    cap = cv2.VideoCapture(filePath)
    # cap.set(3, 1280)
    # cap.set(4, 720)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ## open and set props
    vout = cv2.VideoWriter('results.avi', fourcc, 30, (1920, 1080))
    # vout = cv2.VideoWriter()
    # vout.open('output.mpeg', fourcc, 20, (640, 368), True)
    fpss = []
    model_path = r"E:\zhonghuan_door\train_resnet\model\resnet18_best.pth"
    model = AlexnetInference(model_path)
    while True:
        ret, frame_read = cap.read()
        #frame_read = cv2.resize(frame_read, (608, 608))
        if not ret:
            break
        prev_time = time.time()
        label = model.inference_img(frame_read)
        frame_read = model.draw_result(frame_read, label)
        cv2.imshow('A Gxy App for classification', frame_read)
        vout.write(frame_read)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()
    vout.release()
    fps = np.array(fpss)
    # print('Mean FPS: ', fps.mean())
    # print('Std FPS: ', fps.std())
