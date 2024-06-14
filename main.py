import cv2

def cout(a):
    print("****************************")
    print(a)
    print(type(a))
    print("****************************")


if __name__ == '__main__':
    for i in range (18):
        img = cv2.imread(f"data/emotion_img/images/{i}/Anger.jpg")
        print(img.shape[1]/img.shape[0])

