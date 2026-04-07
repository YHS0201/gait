import os
import cv2
import pickle
import argparse
import numpy as np
from tqdm import tqdm


def letterbox_image(img, target_size, interp=cv2.INTER_LINEAR):
    # target_size: (H, W)
    # Always pad with black borders (zeros) while keeping aspect ratio.
    h, w = img.shape[:2]
    th, tw = target_size
    if h == 0 or w == 0:
        # return black image of target size
        if img.ndim == 3:
            return np.zeros((th, tw, img.shape[2]), dtype=img.dtype)
        return np.zeros((th, tw), dtype=img.dtype)
    scale = min(th / h, tw / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    # create black canvas
    if img.ndim == 3:
        out = np.zeros((th, tw, 3), dtype=img.dtype)
    else:
        out = np.zeros((th, tw), dtype=img.dtype)
    y_offset = (th - nh) // 2
    x_offset = (tw - nw) // 2
    out[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return out


def cut_img_to_center_mask(mask, out_size=(64, 64)):
    # mask: single channel uint8, values arbitrary (0 background)
    # returns uint8 array of shape out_size, or None if invalid
    h, w = mask.shape
    ys = mask.sum(axis=1)
    if ys.max() == 0:
        return None
    y_top = (ys != 0).argmax()
    y_btm = (ys != 0).cumsum().argmax()
    crop = mask[y_top:y_btm+1, :]
    if crop.size == 0:
        return None
    ch = crop.shape[0]
    cw = crop.shape[1]
    th = out_size[0]
    # scale to fixed height th, keep ratio
    scale = th / ch
    nw = int(round(cw * scale))
    if nw <= 0:
        return None
    resized = cv2.resize(crop, (nw, th), interpolation=cv2.INTER_NEAREST)
    # find center by column cumulative
    col_cum = resized.sum(axis=0).cumsum()
    total = resized.sum()
    if total == 0:
        return None
    center = int((col_cum > total / 2).argmax())
    tw = out_size[1]
    half = tw // 2
    left = center - half
    right = center + half
    if left < 0:
        pad = np.zeros((th, -left), dtype=resized.dtype)
        resized = np.concatenate([pad, resized], axis=1)
        left = 0
        right = tw
    if right > resized.shape[1]:
        pad = np.zeros((th, right - resized.shape[1]), dtype=resized.dtype)
        resized = np.concatenate([resized, pad], axis=1)
    out = resized[:, left:right]
    if out.shape[1] != tw:
        # final safety
        out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_NEAREST)
    return out.astype('uint8')


def process_pair(rgb_root, sil_root, out_root, target_rgb=(128, 64), aligned_sil_size=(64, 64)):
    # iterate ids from rgb_root for robust behavior
    for _id in tqdm(sorted(os.listdir(rgb_root)), desc='ids'):
        id_rgb = os.path.join(rgb_root, _id)
        id_sil = os.path.join(sil_root, _id)
        if not os.path.isdir(id_rgb):
            continue
        # types under id
        for _type in sorted(os.listdir(id_rgb)):
            type_rgb = os.path.join(id_rgb, _type)
            type_sil = os.path.join(id_sil, _type) if os.path.isdir(id_sil) else None
            if not os.path.isdir(type_rgb):
                continue
            for _view in sorted(os.listdir(type_rgb)):
                view_rgb = os.path.join(type_rgb, _view)
                view_sil = os.path.join(type_sil, _view) if type_sil and os.path.isdir(type_sil) else None
                if not os.path.isdir(view_rgb):
                    continue
                imgs = []
                sils = []
                ratios = []
                aligned_sils = []
                # list rgb image files
                for img_file in sorted(os.listdir(view_rgb)):
                    img_path = os.path.join(view_rgb, img_file)
                    # corresponding sil path: same filename base + .png
                    base = os.path.splitext(img_file)[0]
                    sil_fname = base + '.png'
                    sil_path = os.path.join(view_sil, sil_fname) if view_sil else None
                    if not sil_path or not os.path.exists(sil_path):
                        # skip frames without mask
                        continue
                    # read
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    sil = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
                    if sil is None:
                        continue
                    h, w = img.shape[:2]
                    ratios.append(float(w) / float(h) if h > 0 else 0.0)
                    # letterbox both to target_rgb size
                    rgb_padded = letterbox_image(img, target_rgb, interp=cv2.INTER_LINEAR)
                    # convert to RGB CHW
                    rgb_padded = cv2.cvtColor(rgb_padded, cv2.COLOR_BGR2RGB)
                    rgb_chw = np.transpose(rgb_padded, (2, 0, 1)).astype('uint8')
                    imgs.append(rgb_chw)
                    # letterbox silhouette as well (nearest)
                    sil_padded = letterbox_image(sil, target_rgb, interp=cv2.INTER_NEAREST)
                    sils.append(sil_padded.astype('uint8'))
                    # aligned small sil using original sil (not padded) to keep proportion
                    aligned = cut_img_to_center_mask(sil, out_size=aligned_sil_size)
                    if aligned is None:
                        # fallback: resize padded sil to aligned size
                        aligned = cv2.resize(sil_padded, (aligned_sil_size[1], aligned_sil_size[0]), interpolation=cv2.INTER_NEAREST)
                    aligned_sils.append(aligned.astype('uint8'))
                if len(imgs) > 0:
                    out_dir = os.path.join(out_root, _id, _type, _view)
                    os.makedirs(out_dir, exist_ok=True)
                    pickle.dump(np.asarray(imgs), open(os.path.join(out_dir, f"{_view}-rgbs.pkl"), 'wb'))
                    pickle.dump(np.asarray(sils), open(os.path.join(out_dir, f"{_view}-sils.pkl"), 'wb'))
                    pickle.dump(np.asarray(ratios), open(os.path.join(out_dir, f"{_view}-ratios.pkl"), 'wb'))
                    pickle.dump(np.asarray(aligned_sils), open(os.path.join(out_dir, f"{_view}-aligned-sils.pkl"), 'wb'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_root', required=True, help='RGB root directory (id/type/view)')
    parser.add_argument('--sil_root', required=True, help='Silhouette root directory (id/type/view)')
    parser.add_argument('--out_root', required=True, help='Output root for pkls')
    parser.add_argument('--rgb_h', type=int, default=128)
    parser.add_argument('--rgb_w', type=int, default=64)
    parser.add_argument('--aligned_h', type=int, default=64)
    parser.add_argument('--aligned_w', type=int, default=64)
    args = parser.parse_args()

    target_rgb = (args.rgb_h, args.rgb_w)
    aligned_sil = (args.aligned_h, args.aligned_w)
    process_pair(args.rgb_root, args.sil_root, args.out_root, target_rgb=target_rgb, aligned_sil_size=aligned_sil)


if __name__ == '__main__':
    main()
