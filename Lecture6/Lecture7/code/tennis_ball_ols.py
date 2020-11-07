import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imsave

# NOTE: The majority of the code in this file was taken from Fall 2019 CS189 HW4

imFile = './images/stpeters_probe_small.png'
compositeFile = './images/tennis.png'
targetFile = './images/interior.jpg'

def loadImages():
    imFile = './images/stpeters_probe_small.png'
    compositeFile = './images/tennis.png'
    targetFile = './images/interior.jpg'
    
    data = imread(imFile).astype('float')*1.5
    tennis = imread(compositeFile).astype('float')
    target = imread(targetFile).astype('float')/255

    return data, tennis, target

def extractNormals(img):

    d = img.shape[0]
    r = d / 2
    ns = []
    vs = []
    for i in range(d):
        for j in range(d):

            x = j - r
            y = i - r
            if x*x + y*y > r*r-100:
                continue

            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            view = np.asarray([0,0,-1])
            n = 2*n*(np.sum(n*view))-view
            ns.append(n)
            vs.append(img[i,j])

    return np.asarray(ns), np.asarray(vs)

def renderSphere(r,coeff):

    d = 2*r
    img = -np.ones((d,d,3))
    ns = []
    ps = []

    for i in range(d):
        for j in range(d):

            x = j - r
            y = i - r
            if x*x + y*y > r*r:
                continue

            z = np.sqrt(r*r-x*x-y*y)
            n = np.asarray([x,y,z])
            n = n / np.sqrt(np.sum(np.square(n)))
            ns.append(n)
            ps.append((i,j))

    ns = np.asarray(ns)
    B = computeBasis(ns)
    vs = B.dot(coeff)

    for p,v in zip(ps,vs):
        img[p[0],p[1]] = np.clip(v,0,255)

    return img

def relightSphere(img, coeff):
    img = renderSphere(int(img.shape[0]/2),coeff)/255*img/255
    return img

def compositeImages(source, target):
    
    out = target.copy()
    cx = int(target.shape[1]/2)
    cy = int(target.shape[0]/2)
    sx = cx - int(source.shape[1]/2)
    sy = cy - int(source.shape[0]/2)

    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            if np.sum(source[i,j]) >= 0:
                out[sy+i,sx+j] = source[i,j]

    return out

def computeBasis(ns):

    B = []
    
    for n in ns:
        l_1 = 1
        l_2 = n[1]
        l_3 = n[0]
        l_4 = n[2]
        l_5 = n[0] * n[1]
        l_6 = n[1] * n[2]
        l_7 = 3*(n[2]**2) - 1
        l_8 = n[0] * n[2]
        l_9 = n[0]**2 - n[1]**2
        row_n = [l_1, l_2, l_3, l_4, l_5, l_6, l_7, l_8, l_9]
        B.append(row_n)
 
    return np.asmatrix(B)
    
if __name__ == '__main__':

    data, tennis, target = loadImages()
    ns, vs = extractNormals(data)
    B = computeBasis(ns)

    Bp = B[::50]
    vsp = vs[::50]
    
    solver = input('Original image or use OLS to adjust lighting?\nType "original" or "OLS":\n')
    if solver == 'OLS':
        coeff = np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B) @ vs
        img = relightSphere(tennis, coeff)
        output = compositeImages(img, target)
        #print('Coefficients:\n'+str(coeff))
        fig, ax = plt.subplots(figsize=(12,8))
        plt.figure(1)
        plt.imshow(output)
        plt.show()
    elif solver == 'original':
        coeff = np.zeros((9,3))
        coeff[0,:] = 255
        img = relightSphere(tennis, coeff)
        output = compositeImages(img, target)
        #print('Coefficients:\n'+str(coeff))
        fig, ax = plt.subplots(figsize=(12,8))
        plt.figure(1)
        plt.imshow(output)
        plt.show()
    else:
        print("Invalid input")
        exit(1)
