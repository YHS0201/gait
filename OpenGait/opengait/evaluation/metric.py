import torch
import numpy as np
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from utils import is_tensor


def cuda_dist(x, y, metric='euc'):  # official
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    # num_bin = 12# x.size(2)
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            _dist = torch.sum(_x**2, 1).unsqueeze(1) + torch.sum(
                _y**2,
                1).unsqueeze(0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin


def euc_dist(x, y):
    x = x.reshape(-1, 1024)
    y = y.reshape(-1, 1024)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    dist = []
    for ux in x:
        udist = np.linalg.norm(ux - y, axis=1)
        dist.append(udist)
    dist = np.array(dist)
    print(dist.shape)
    return dist


def cuda_dist_flatten(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    x_flatten = x.view(x.size(0), -1)
    y_flatten = y.view(y.size(0), -1)
    if metric == 'cos':
        dist = 1 - torch.matmul(x_flatten, y_flatten.transpose(0, 1))
    else:
        dist = torch.cdist(x_flatten, y_flatten, p=2)
    return dist


def cuda_dist_torch(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # n c p
        y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            dist += torch.cdist(_x, _y, p=2)
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin


def cuda_dist_norm(x, y, metric='euc'):
    x = torch.from_numpy(x).cuda()
    y = torch.from_numpy(y).cuda()

    x = F.normalize(x, p=2, dim=1)  # n c p
    y = F.normalize(y, p=2, dim=1)  # n c p
    num_bin = x.size(2)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y).cuda()
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))
        else:
            # _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(
            #     0) - 2 * torch.matmul(_x, _y.transpose(0, 1))
            # dist += torch.sqrt(F.relu(_dist))
            dist += torch.cdist(_x, _y, p=2)
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin


def cuda_dist_all(x, y, metric='euc'):
    # calculate distance between all pairs of x and y using different methods
    if metric != 'all':
        dist_official = cuda_dist(x, y, metric)
        dist_flatten = cuda_dist_flatten(x, y, metric)
        dist_torch = cuda_dist_torch(x, y, metric)
        dist_all = {
            "official": dist_official,
            "flatten": dist_flatten,
            "torch": dist_torch
        }
    if metric == 'all':
        dist_official = cuda_dist(x, y, 'euc')
        dist_flatten = cuda_dist_flatten(x, y, 'euc')
        dist_torch = cuda_dist_torch(x, y, 'euc')
        dist_cos_official = cuda_dist(x, y, 'cos')
        dist_cos_flatten = cuda_dist_flatten(x, y, 'cos')
        # dist_cos_torch code is same as dis_cos_official
        dist_official_norm = cuda_dist_norm(x, y, 'euc')
        dist_cos_official_norm = cuda_dist_norm(x, y, 'cos')
        dist_all = {
            "official": dist_official,
            "flatten": dist_flatten,
            "torch": dist_torch,
            "cos_official": dist_cos_official,
            "cos_flatten": dist_cos_flatten,
            "dist_official_norm": dist_official_norm,
            "dist_cos_official_norm": dist_cos_official_norm
        }
    return dist_all


def mean_iou(msk1, msk2, eps=1.0e-9):
    if not is_tensor(msk1):
        msk1 = torch.from_numpy(msk1).cuda()
    if not is_tensor(msk2):
        msk2 = torch.from_numpy(msk2).cuda()
    n = msk1.size(0)
    inter = msk1 * msk2
    union = ((msk1 + msk2) > 0.).float()
    miou = inter.view(n, -1).sum(-1) / (union.view(n, -1).sum(-1) + eps)
    return miou


def compute_ACC_mAP(distmat,
                    q_pids,
                    g_pids,
                    q_views=None,
                    g_views=None,
                    rank=1):
    num_q, _ = distmat.shape
    # indices = np.argsort(distmat, axis=1)
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_ACC = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        q_idx_dist = distmat[q_idx]
        q_idx_glabels = g_pids
        if q_views is not None and g_views is not None:
            q_idx_mask = np.isin(g_views, q_views[q_idx],
                                 invert=True) | np.isin(
                                     g_pids, q_pids[q_idx], invert=True)
            q_idx_dist = q_idx_dist[q_idx_mask]
            q_idx_glabels = q_idx_glabels[q_idx_mask]

        assert (len(q_idx_glabels) >
                0), "No gallery after excluding identical-view cases!"
        q_idx_indices = np.argsort(q_idx_dist)
        q_idx_matches = (q_idx_glabels[q_idx_indices] == q_pids[q_idx]).astype(
            np.int32)

        # binary vector, positions with value 1 are correct matches
        # orig_cmc = matches[q_idx]
        orig_cmc = q_idx_matches
        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_ACC.append(cmc[rank - 1])

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()

        if num_rel > 0:
            num_valid_q += 1.
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

    # all_ACC = np.asarray(all_ACC).astype(np.float32)
    ACC = np.mean(all_ACC)
    mAP = np.mean(all_AP)

    return ACC, mAP


def evaluate_rank(distmat, p_lbls, g_lbls, max_rank=50):
    '''
    Copy from https://github.com/Gait3D/Gait3D-Benchmark/blob/72beab994c137b902d826f4b9f9e95b107bebd78/lib/utils/rank.py#L12-L63
    '''
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print('Note: number of gallery samples is quite small, got {}'.format(
            num_g))
    print(
        f'Note, number of gallery samples is {num_g} and number of probe samples is {num_p}'
    )
    indices = np.argsort(distmat, axis=1)

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)
    # 保存matches
    np.save('matches.npy', matches)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)  # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_rank_all_analysis(distmat, p_lbls, g_lbls, probe_rel_paths, gallery_rel_paths, msg_mgr, max_rank=50, max_num=10):
    '''
    对所有场景一起进行分析，分场景不要用
    '''
    num_p, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        msg_mgr.log_info('Note: number of gallery samples is quite small, got {}'.format(
            num_g))
    msg_mgr.log_info(
        f'Note, number of gallery samples is {num_g} and number of probe samples is {num_p}'
    )
    indices = np.argsort(distmat, axis=1)

    sorted_rel_paths_mat = gallery_rel_paths[indices]

    matches = (g_lbls[indices] == p_lbls[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each probe
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_p = 0.  # number of valid probe

    for p_idx in range(num_p):
        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[p_idx]
        if not np.any(raw_cmc):
            # this condition is true when probe identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        pos_idx = np.where(raw_cmc == 1)  # 返回坐标，此处raw_cmc为一维矩阵，所以返回相当于index
        max_pos_idx = np.max(pos_idx)
        neg_idx = np.where(raw_cmc == 0) if not np.all(raw_cmc == 1) else -1
        if not neg_idx == -1:
            min_neg_idx = np.min(neg_idx)
            if min_neg_idx < max_pos_idx:
                msg_mgr.log_info(f"For probe '{probe_rel_paths[p_idx]}', there exists hard negative samples:(")
                hard_neg_indices = np.where(raw_cmc[min_neg_idx:max_pos_idx] == 0)[0] + min_neg_idx
                if len(hard_neg_indices) < max_num:
                    max_num = len(hard_neg_indices)
                msg_mgr.log_info(f'Top{max_num} hard negative samples are:')
                for i in hard_neg_indices[:max_num]:
                    msg_mgr.log_info(f"'{sorted_rel_paths_mat[p_idx][i]}'")
                fn_indices = np.where(raw_cmc[min_neg_idx:max_pos_idx] == 1)[0] + min_neg_idx
                max_num = 10
                if len(fn_indices) < max_num:
                    max_num = len(fn_indices)
                msg_mgr.log_info(f'Top{max_num} smallest positive samples are:')
                for i in fn_indices[:-max_num-1:-1]:
                    msg_mgr.log_info(f"'{sorted_rel_paths_mat[p_idx][i]}'")

        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_p += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_p > 0, 'Error: all probe identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_p

    return all_cmc, all_AP, all_INP


def evaluate_many(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(
            num_g))
    indices = np.argsort(distmat, axis=1)  # 对应位置变成从小到大的序号
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(
        np.int32)  # 根据indices调整顺序 g_pids[indices]
    # print(matches)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP
