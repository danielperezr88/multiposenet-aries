import os
import math
import json
import numpy as np
from tqdm import tqdm
from random import shuffle

from pycocotools.cocoeval import COCOeval

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

    x1, x2, y1, y2 = (int(x1), int(x2), int(y1), int(y2))

    return img[x1:x2, y1:y2]

def Evaluation(model,optin,coco):
    print ('------------Evaulation Started------------')
    coeff = optin.coeff
    in_thres = optin.threshold
    n_kernel = optin.window_size
    modelname = 'temporary'

    cocodir = 'data/annotations/person_keypoints_val2017.json'
    ann = json.load(open(cocodir))
    bbox_results = ann['annotations']

    img_ids = coco.getImgIds(catIds=[1])

    peak_results = []

    for i in img_ids:
        anns = coco.loadAnns(coco.getAnnIds(imgIds=i))
        kps = [a['keypoints'] for a in anns]

        idx = 0

        ks = []
        for i in range(17):
            t = []
            for k in kps:
                x = k[0::3][i]
                y = k[1::3][i]
                v = k[2::3][i]

                if v > 0:
                    t.append([x, y, 1, idx])
                    idx += 1
            ks.append(t)
        image_id = anns[0]['image_id']
        peaks = ks

        element = {
            'image_id': image_id,
            'peaks': peaks,
            'file_name': coco.loadImgs(image_id)[0]['file_name']
        }

        peak_results.append(element)

    shuffle(peak_results)

    my_results = []
    image_ids = []

    w = int(18 * coeff)
    h = int(28 * coeff)

    temporary_peak_res = []
    for p in peak_results:
        if (sum(1 for i in p['peaks'] if i != []) >= 0):
            temporary_peak_res.append(p)
    peak_results = temporary_peak_res

    for p in tqdm(peak_results):
        idx = p['image_id']
        image_ids.append(idx)

        peaks = p['peaks']
        bboxes = [k['bbox'] for k in bbox_results if k['image_id'] == idx]


        if len(bboxes) == 0 or len(peaks) == 0:
            continue

        weights_bbox = np.zeros((len(bboxes), h, w, 4, 17))

        for joint_id, peak in enumerate(peaks):


            for instance_id, instance in enumerate(peak):

                p_x = instance[0]
                p_y = instance[1]

                for bbox_id, b in enumerate(bboxes):

                    is_inside = p_x > b[0] - b[2] * in_thres and \
                                p_y > b[1] - b[3] * in_thres and \
                                p_x < b[0] + b[2] * (1.0 + in_thres) and \
                                p_y < b[1] + b[3] * (1.0 + in_thres)

                    if is_inside:
                        x_scale = float(w) / math.ceil(b[2])
                        y_scale = float(h) / math.ceil(b[3])

                        x0 = int((p_x - b[0]) * x_scale)
                        y0 = int((p_y - b[1]) * y_scale)

                        if x0 >= w and y0 >= h:
                            x0 = w - 1
                            y0 = h - 1
                        elif x0 >= w:
                            x0 = w - 1
                        elif y0 >= h:
                            y0 = h - 1
                        elif x0 < 0 and y0 < 0:
                            x0 = 0
                            y0 = 0
                        elif x0 < 0:
                            x0 = 0
                        elif y0 < 0:
                            y0 = 0

                        p = 1e-9

                        weights_bbox[bbox_id, y0, x0, :, joint_id] = [1, instance[2], instance[3], p]

        old_weights_bbox = np.copy(weights_bbox)

        for j in range(weights_bbox.shape[0]):
            for t in range(17):
                weights_bbox[j, :, :, 0, t] = gaussian(weights_bbox[j, :, :, 0, t])
            # weights_bbox[j, :, :, 0, :]      = gaussian_multi_input_mp(weights_bbox[j, :, :, 0, :])

        output_bbox = []
        for j in range(weights_bbox.shape[0]):
            inp = weights_bbox[j, :, :, 0, :]
            output = model.predict(np.expand_dims(inp, axis=0))
            output_bbox.append(output[0])

        output_bbox = np.array(output_bbox)

        keypoints_score = []

        for t in range(17):
            indexes = np.argwhere(old_weights_bbox[:, :, :, 0, t] == 1)
            keypoint = []
            for i in indexes:
                cr = crop(output_bbox[i[0], :, :, t], (i[1], i[2]), N=n_kernel)
                score = np.sum(cr)

                kp_id = old_weights_bbox[i[0], i[1], i[2], 2, t]
                kp_score = old_weights_bbox[i[0], i[1], i[2], 1, t]
                p_score = old_weights_bbox[i[0], i[1], i[2], 3, t]  ## ??
                bbox_id = i[0]

                score = kp_score * score

                s = [kp_id, bbox_id, kp_score, score]

                keypoint.append(s)
            keypoints_score.append(keypoint)

        bbox_keypoints = np.zeros((weights_bbox.shape[0], 17, 3))
        bbox_ids = np.arange(len(bboxes)).tolist()

        # kp_id, bbox_id, kp_score, my_score
        for i in range(17):
            joint_keypoints = keypoints_score[i]
            if len(joint_keypoints) > 0:

                kp_ids = list(set([x[0] for x in joint_keypoints]))

                table = np.zeros((len(bbox_ids), len(kp_ids), 4))

                for b_id, bbox in enumerate(bbox_ids):
                    for k_id, kp in enumerate(kp_ids):
                        own = [x for x in joint_keypoints if x[0] == kp and x[1] == bbox]

                        if len(own) > 0:
                            table[bbox, k_id] = own[0]
                        else:
                            table[bbox, k_id] = [0] * 4

                for b_id, bbox in enumerate(bbox_ids):

                    row = np.argsort(-table[bbox, :, 3])

                    if table[bbox, row[0], 3] > 0:
                        for r in row:
                            if table[bbox, r, 3] > 0:
                                column = np.argsort(-table[:, r, 3])

                                if bbox == column[0]:
                                    bbox_keypoints[bbox, i, :] = [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                    break
                                else:
                                    row2 = np.argsort(table[column[0], :, 3])
                                    if row2[0] == r:
                                        bbox_keypoints[bbox, i, :] = \
                                        [x[:3] for x in peaks[i] if x[3] == table[bbox, r, 0]][0]
                                        break
            else:
                for j in range(weights_bbox.shape[0]):
                    b = bboxes[j]
                    x_scale = float(w) / math.ceil(b[2])
                    y_scale = float(h) / math.ceil(b[3])

                    for t in range(17):
                        indexes = np.argwhere(old_weights_bbox[j, :, :, 0, t] == 1)
                        if len(indexes) == 0:
                            max_index = np.argwhere(output_bbox[j, :, :, t] == np.max(output_bbox[j, :, :, t]))
                            bbox_keypoints[j, t, :] = [max_index[0][1] / x_scale + b[0],
                                                       max_index[0][0] / y_scale + b[1], 0]

        my_keypoints = []

        for i in range(bbox_keypoints.shape[0]):
            k = np.zeros(51)
            k[0::3] = bbox_keypoints[i, :, 0]
            k[1::3] = bbox_keypoints[i, :, 1]
            k[2::3] = [2] * 17

            pose_score = 0
            count = 0
            for f in range(17):
                if bbox_keypoints[i, f, 0] != 0 and bbox_keypoints[i, f, 1] != 0:
                    count += 1
                pose_score += bbox_keypoints[i, f, 2]
            pose_score /= 17.0

            my_keypoints.append(k)

            image_data = {
                'image_id': idx,
                'bbox': bboxes[i],
                'score': pose_score,
                'category_id': 1,
                'keypoints': k.tolist()
            }
            my_results.append(image_data)


    ann_filename = 'data/val2017_PRN_keypoint_results_{}.json'.format(modelname)
    # write output
    json.dump(my_results, open(ann_filename, 'w'), indent=4)

    # load results in COCO evaluation tool
    coco_pred = coco.loadRes(ann_filename)

    # run COCO evaluation
    coco_eval = COCOeval(coco, coco_pred, 'keypoints')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    os.remove(ann_filename)
