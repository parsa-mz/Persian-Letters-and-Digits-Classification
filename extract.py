import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


chars1 = ['0', '1', 'a', 'b', 'p', 't', 's_3noghte', 'j', 'ch', 'h_jimi', 'kh', 'd', 
        'dz', 'r', 'rz', 'zh', 's', 'sh', 'sad', '2', '3']

chars2 = ['4', '5', 'zad', 'ta', 'za', 'ain', 'ghain', 'f', 'gh', 'k', 'g', 'l',
        'm', 'n', 'v', 'h', 'y', '6', '7', '8', '9']


## acuro marker detection
def Aruco(I):
    G = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    ## acuro 
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    parameters =  cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(G, aruco_dict, parameters=parameters)
    frame_markers = cv2.aruco.drawDetectedMarkers(I.copy(), corners, ids)

    # for temporary saving points
    points = [0, 0, 0, 0]

    plt.figure('Aruco symbol extraction')
    plt.imshow(frame_markers)

    for i in range(len(ids)):
        # for each symbol get it's 4 corners
        c = corners[i][0]

        # id[33]: top-left    - pos=0      id[31]: top-right    - pos=1
        # id[32]: buttom-left - pos=3      id[30]: buttom-right - pos=2

        if ids[i] == 32:
            x, y = c[2, 0], c[2, 1]
            points[3] = (x, y)
        if ids[i] == 33:
            x, y = c[3, 0], c[3, 1]
            points[2] = (x, y)
        if ids[i] == 31:
            x, y = c[0, 0], c[0, 1]
            points[1] = (x, y)
        if ids[i] == 30:
            x, y = c[1, 0], c[1, 1]
            points[0] = (x, y)
        
        plt.plot(x, y, "o", label = "id={0}".format(ids[i]))

    plt.legend()    
    return points


## get pespective
def Perspective_Transform(I, p):
    n, m = 300, 630
    output_size = (n,m)

    points1 = np.array([p[0],p[1],p[2],p[3]], dtype=np.float32)
    points2 = np.array([[0, 0], [n, 0], [n, m], [0, m]], dtype=np.float32)

    I_copy = np.array(I)

    M = cv2.getPerspectiveTransform(points1, points2)
    J = cv2.warpPerspective(I_copy, M, output_size)

    plt.figure('Perspective Transform')
    plt.imshow(J)
    #plt.show()
    return J
    

# save images
def SaveImages(J, dataset_type):
    if dataset_type == '1':
        chars = chars1
    else:
        chars = chars2
    j = 0
    for y in range(0, 630, 30):
        index = len(glob.glob("dataset\\train\\" + chars[j] + "/*")) + 1
        test_index = len(glob.glob("dataset\\test\\" + chars[j] + "/*")) + 1
        for x in range(0, 300, 30):
            crop_img = J[y:y+30, x:x+30]
            crop_img = crop_img[2:28, 2:28]
            if x == 270 or x == 240:
                filename = "dataset\\test\\" + chars[j] + "\\" + str(test_index) + ".jpg"
                test_index += 1
            else:
                filename = "dataset\\train\\" + chars[j] + "\\" + str(index) + ".jpg"
                index += 1
            cv2.imwrite(filename, crop_img)
        j += 1


def CreateDirs():
    for ch in chars2:
        os.mkdir("dataset\\train\\" + ch)
        os.mkdir("dataset\\test\\" + ch)
    for ch in chars1:
        os.mkdir("dataset\\train\\" + ch)
        os.mkdir("dataset\\test\\" + ch)


if __name__ == "__main__":
    files = glob.glob("data\*.jpg")
    
    for f in files:
        filename = f.split('\\')[-1]
        try:
            I = cv2.imread(f)
            # find aruco symbols positions
            points1 = Aruco(I)
            # align image 
            J = Perspective_Transform(I, points1)
            # save output images
            SaveImages(J, filename[8])
            print("extracted", filename)
        except KeyboardInterrupt:
            exit()
        except:
            print('error extracting ', filename)