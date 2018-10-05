import math
import numpy as np
from random import shuffle
from skimage.filters import gaussian

sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])

def multivariate_gaussian(N, sigma=2):
    t = 4
    X = np.linspace(-t, t, N)
    Y = np.linspace(-t, t, N)
    X, Y = np.meshgrid(X, Y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    mu = np.array([0., 0.])
    sigma = np.array([[sigma, 0], [0, sigma]])
    n = mu.shape[0]
    Sigma_det = np.linalg.det(sigma)
    Sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
    return np.exp(-fac / 2) / N


def crop_paste(img, c, N=13, sigma=2):
    Z = multivariate_gaussian(N, sigma)

    H = img.shape[1]
    W = img.shape[0]

    h = (Z.shape[0] - 1) / 2

    N = Z.shape[0]
    x1 = (c[0] - h)
    y1 = (c[1] - h)

    x2 = (c[0] + h) + 1
    y2 = (c[1] + h) + 1

    zx1 = 0
    zy1 = 0
    zx2 = N + 1
    zy2 = N + 1

    if x1 < 0:
        x1 = 0
        zx1 = 0 - (c[0] - h)

    if y1 < 0:
        y1 = 0
        zy1 = 0 - (c[1] - h)

    if x2 > W - 1:
        x2 = W - 1
        zx2 = x2 - x1 + 1
        x2 = W

    if y2 > H - 1:
        y2 = H - 1
        zy2 = y2 - y1 + 1
        y2 = H

    img[x1:x2, y1:y2] = np.maximum(Z[zx1:zx2, zy1:zy2], img[x1:x2, y1:y2])


'''
def gaussian(img, N = 13, sigma=2):
    cs = np.where(img==1)
    img = np.zeros_like(img)
    for c in zip(cs[0], cs[1]):
        crop_paste(img, c, N, sigma)
    return img
'''


def gaussian_multi_input_mp(inp):
    '''
    :param inp: Multi person ground truth heatmap input (17 ch) Each channel contains multiple joints.
    :return: out: Gaussian augmented output. Values are between 0. and 1.
    '''

    h, w, ch = inp.shape
    out = np.zeros_like(inp)
    for i in range(ch):
        layer = inp[:, :, i]
        ind = np.argwhere(layer == 1)
        b = []
        if len(ind) > 0:
            for j in ind:
                t = np.zeros((h, w))
                t[j[0], j[1]] = 1
                t = gaussian(t, sigma=sigmas[i], mode='constant')
                t = t * (1 / t.max())
                b.append(t)

            out[:, :, i] = np.maximum.reduce(b)
        else:
            out[:, :, i] = np.zeros((h, w))
    return out


def gaussian_multi_output(inp):
    '''
    :param inp: Single person ground truth heatmap input (17 ch) Each channel contains one joint.
    :return: out: Gaussian augmented output. Values are between 0. and 1.
    '''
    h, w, ch = inp.shape
    out = np.zeros_like(inp)
    for i in range(ch):
        j = np.argwhere(inp[:, :, i] == 1)
        if len(j) == 0:
            out[:, :, i] = np.zeros((h, w))
            continue
        j = j[0]
        t = np.zeros((h, w))
        t[j[0], j[1]] = 1
        t = gaussian(t, sigma=sigmas[i], mode='constant')
        out[:, :, i] = t * (1 / t.max())
    return out


def crop(img, c, N=13):
    H = img.shape[1]
    W = img.shape[0]

    h = (N - 1) / 2

    x1 = (c[0] - h)
    y1 = (c[1] - h)

    x2 = (c[0] + h) + 1
    y2 = (c[1] + h) + 1

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x2 > W - 1:
        x2 = W

    if y2 > H - 1:
        y2 = H

    return img[x1:x2, y1:y2]

def get_data(ann_data, coco, height, width,thres):
    weights = np.zeros((height, width, 17))
    output = np.zeros((height, width, 17))


    bbox = ann_data['bbox']
    x = int(bbox[0])
    y = int(bbox[1])
    w = float(bbox[2])
    h = float(bbox[3])

    x_scale = float(width) / math.ceil(w)
    y_scale = float(height) / math.ceil(h)

    kpx = ann_data['keypoints'][0::3]
    kpy = ann_data['keypoints'][1::3]
    kpv = ann_data['keypoints'][2::3]

    for j in range(17):
        if kpv[j] > 0:
            x0 = int((kpx[j] - x) * x_scale)
            y0 = int((kpy[j] - y) * y_scale)

            if x0 >= width and y0 >= height:
                output[height - 1, width - 1, j] = 1
            elif x0 >= width:
                output[y0, width - 1, j] = 1
            elif y0 >= height:
                output[height - 1, x0, j] = 1
            elif x0 < 0 and y0 < 0:
                output[0, 0, j] = 1
            elif x0 < 0:
                output[y0, 0, j] = 1
            elif y0 < 0:
                output[0, x0, j] = 1
            else:
                output[y0, x0, j] = 1

    img_id = ann_data['image_id']
    img_data = coco.loadImgs(img_id)[0]
    ann_data = coco.loadAnns(coco.getAnnIds(img_data['id']))

    for ann in ann_data:
        kpx = ann['keypoints'][0::3]
        kpy = ann['keypoints'][1::3]
        kpv = ann['keypoints'][2::3]

        for j in range(17):
            if kpv[j] > 0:
                if (kpx[j] > bbox[0] - bbox[2] * thres and kpx[j] < bbox[0] + bbox[2] * (1 + thres)):
                    if (kpy[j] > bbox[1] - bbox[3] * thres and kpy[j] < bbox[1] + bbox[3] * (1 + thres)):
                        x0 = int((kpx[j] - x) * x_scale)
                        y0 = int((kpy[j] - y) * y_scale)

                        if x0 >= width and y0 >= height:
                            weights[height - 1, width - 1, j] = 1
                        elif x0 >= width:
                            weights[y0, width - 1, j] = 1
                        elif y0 >= height:
                            weights[height - 1, x0, j] = 1
                        elif x0 < 0 and y0 < 0:
                            weights[0, 0, j] = 1
                        elif x0 < 0:
                            weights[y0, 0, j] = 1
                        elif y0 < 0:
                            weights[0, x0, j] = 1
                        else:
                            weights[y0, x0, j] = 1

    for t in range(17):
        weights[:, :, t] = gaussian(weights[:, :, t])
    output  =  gaussian(output, sigma=2, mode='constant', multichannel=True)
    #weights = gaussian_multi_input_mp(weights)
    return weights, output


def get_anns(coco):
    '''
    :param coco: COCO instance
    :return: anns: List of annotations that contain person with at least 6 keypoints
    '''
    ann_ids = coco.getAnnIds()
    anns = []
    for i in ann_ids:
        ann = coco.loadAnns(i)[0]
        if ann['iscrowd'] == 0 and ann['num_keypoints'] > 4:
            anns.append(ann) # ann
    sorted_list = sorted(anns, key=lambda k: k['num_keypoints'], reverse=True)
    return sorted_list


def train_bbox_generator(coco_train,batch_size,height,width,thres):
    anns = get_anns(coco_train)
    while 1:
        shuffle(anns)
        for i in range(0, len(anns) // batch_size, batch_size):
            X = np.zeros((batch_size, height, width, 17))
            Y = np.zeros((batch_size, height, width, 17))
            for j in range(batch_size):
                ann_data = anns[i+j]
                try:
                    x, y = get_data(ann_data, coco_train, height, width, thres)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y




def val_bbox_generator(coco_val, batch_size,height,width,thres):
    ann_ids = coco_val.getAnnIds()
    while 1:
        shuffle(ann_ids)
        for i in range(len(ann_ids) // batch_size):
            X = np.zeros((batch_size, height, width, 17))
            Y = np.zeros((batch_size, height, width, 17))
            for j in range(batch_size):
                ann_data = coco_val.loadAnns(ann_ids[i + j])[0]
                try:
                    x, y = get_data(ann_data, coco_val,height,width,thres)
                except:
                    continue
                X[j, :, :, :] = x
                Y[j, :, :, :] = y
            yield X, Y
