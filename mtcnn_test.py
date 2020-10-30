import cv2
import time
import numpy as np
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector

pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)

while True:
    ret, frame = capture.read()
    if ret is True:
        frame = cv2.flip(frame, 1)  # cv2.flip 图像翻转
        s_t = time.time()
        bboxs, landmarks = mtcnn_detector.detect_face(frame)
        for box, marks in zip(bboxs, landmarks):
            frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            for i in range(5):
                frame = cv2.circle(frame, (int(marks[i*2]), int(marks[i*2+1])), 1, (255, 0, 0), 4)
        print(f'time:{time.time()-s_t}')
        cv2.imshow("frame", frame)
        c = cv2.waitKey(10)
        if c == 27:
            break
capture.release()
cv2.destroyAllWindows()  # 关闭窗口
