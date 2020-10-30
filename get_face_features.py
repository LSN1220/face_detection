import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
# from .utils.generate_csv import write_Users_csv,write_Name_csv,read_csv,
#
#
# def prepare_openface(useCuda=False, gpuDevice=0, useMultiGPU=False):
#     model = netOpenFace(useCuda, gpuDevice)
#     model.load_state_dict(torch.load('./models/openface_20180119.pth',map_location=lambda storage, loc: storage))
#
#     if useMultiGPU:
#         model = nn.DataParallel(model)
#
#     return model


img = cv2.imread('lvsongnan .jpg')
H, W, C = img.shape
img = cv2.resize(img, (W//4, H//4))
FrameSize = (img.shape[1], img.shape[0])

pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

bboxs, landmarks = mtcnn_detector.detect_face(img)
for box, marks in zip(bboxs, landmarks):
    img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
    for i in range(5):
        frame = cv2.circle(img, (int(marks[i * 2]), int(marks[i * 2 + 1])), 1, (255, 0, 0), 4)
cv2.imshow("frame", img)
cv2.waitKey(0)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# facenet = prepare_openface()