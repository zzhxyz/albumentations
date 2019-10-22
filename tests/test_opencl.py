import cv2
import numpy as np

import albumentations as A


def test_opencl():
    aug_cpu = A.Compose([
        A.RandomSizedCrop((224, 384), 256, 256, p=1),
        A.ShiftScaleRotate(p=1),
        A.GaussianBlur(p=1),
        # A.ElasticTransform()
    ])

    aug_gpu = A.Compose([
        A.ToOpenCL(),
        A.RandomSizedCrop((224, 384), 256, 256, p=1),
        A.ShiftScaleRotate(p=1),
        A.GaussianBlur(p=1),
        # A.ElasticTransform(),
        A.ToNumpy()
    ])


    print(cv2.getBuildInformation())

    for j in range(16):
        time1 = 0
        time2 = 0

        for i in range(512):
            img = np.zeros((512, 512, 3), dtype=np.uint8)
            # msk = np.zeros((1024, 1024, 3), dtype=np.uint8)

            start1 = cv2.getTickCount()
            aug_cpu(image=img)
            end1 = cv2.getTickCount()

            start2 = cv2.getTickCount()
            aug_gpu(image=img)
            end2 = cv2.getTickCount()

            time1 += end1 - start1
            time2 += end2 - start2

        f = cv2.getTickFrequency()
        print('CPU', time1 / f,
              'OpenCL', time2 / f,
              'Speedup', time1 / time2)
