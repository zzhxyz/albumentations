import matplotlib.pyplot as plt
import cv2
import albumentations as A

def main():

    def get_values(sched, steps=None):
        values = []
        sched.reset()

        if steps is not None:
            rng = range(steps)
        else:
            rng = range(len(sched))

        for _ in rng:
            values.append(sched.value())
            sched.step()

        return values

    lin = A.LinearSchedule(0, 2, a_min=0, a_max=50, steps=100)
    cos = A.CosineRestartsDecay(50, 20, a_min=1, steps=100)
    exp = A.ExpDecay(50, 0.01, a_min=1, steps=100)

    plt.figure()
    plt.plot(get_values(lin), label='Linear')
    plt.plot(get_values(cos), label='Exp')
    plt.plot(get_values(exp), label='Exp')

    concat = A.ConcatSchedule([lin, cos, exp])
    plt.figure()
    plt.plot(get_values(concat, 300), label='Concat')

    plt.show()

    aug = A.Compose([
        A.RandomBrightnessContrast(),
    ], p=concat)

    image = cv2.imread('images/image_1.jpg')

    aug.reset()
    for epoch in range(300):
        aug.step()
        data = aug(image=image)
        cv2.imshow("Image", data['image'])
        cv2.waitKey(25)

if __name__ == '__main__':
    main()