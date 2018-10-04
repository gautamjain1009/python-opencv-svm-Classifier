# encoding: utf-8



# 7 检测  核心：create Hog -》 myDetect—》array-》
# resultArray-》resultArray = -1*alphaArray*supportVArray
# rho-》svm-〉svm.train
import cv2
import numpy as np


featureNum = int(((128 - 16) / 8 + 1) * ((64 - 16) / 8 + 1) * 4 * 9)  # 3780
svm = cv2.ml.SVM_load("svmtest.mat")
alpha = np.zeros((1), np.float32)
rho = svm.getDecisionFunction(0, alpha)
print(rho)
print(alpha)
alphaArray = np.zeros((1, 1), np.float32)
supportVArray = np.zeros((1, featureNum), np.float32)
resultArray = np.zeros((1, featureNum), np.float32)
alphaArray[0, 0] = alpha
resultArray = -1 * alphaArray * supportVArray
# detect
myDetect = np.zeros((3781), np.float32)
for i in range(0, 3780):
    myDetect[i] = resultArray[0, i]
myDetect[3780] = rho[0]

# rho svm （判决）
myHog = cv2.HOGDescriptor()
myHog.setSVMDetector(myDetect)
# load
imageSrc = cv2.imread('1.jpg',1)
# (8,8) win
print(imageSrc.shape)
objs = myHog.detectMultiScale(imageSrc, 0, (8, 8), (32, 32), 1.05, 2)
# xy wh 三维 最后一维
print(objs)
x = int(objs[0][0][0])
y = int(objs[0][0][1])
w = int(objs[0][0][2])
h = int(objs[0][0][3])
# 绘制展示
cv2.rectangle(imageSrc, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('dst', imageSrc)
cv2.waitKey(0)