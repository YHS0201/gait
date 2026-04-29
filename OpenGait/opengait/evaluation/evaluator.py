import os
from time import strftime, localtime
import numpy as np
from utils import get_msg_mgr, mkdir

from .metric import mean_iou, cuda_dist, compute_ACC_mAP, evaluate_rank, evaluate_many, cuda_dist_all, euc_dist, evaluate_rank_all_analysis
from .re_rank import re_ranking
import json
from prettytable import PrettyTable, ALL
import csv
import torch.nn.functional as F
from .evaluator_reduce import evaluate_reduce_target
# from .evaluator_reduce import evaluate_reduce_target
# from .evaluator_reduce_v12 import evaluate_reduce_target_v12
# from .evaluator_reduce_rerank import evaluate_reduce_target_rerank

def save_feature(data, *args, **kwargs):
    features, label, seq_type, view = data['embeddings'], data['labels'], data[
        'types'], data['views']

    save_root = '/mnt/wuyx_data/tmp/save_npy'

    # features = features[:, :, ::4]  # for 16 debug

    for i, one_fea in enumerate(features):
        one_fea = one_fea.reshape(-1)
        save_file = os.path.join(save_root, label[i], seq_type[i], view[i],
                                 view[i] + '.npy')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        np.save(save_file, one_fea)
    return None, None, None, None


def save_feature_for_cargait(data, *args, **kwargs):
    features, labels = data['embeddings'], data['labels']

    save_root = './features'
    os.makedirs(save_root, exist_ok=True)
    save_name = 'train_feats.npz'
    save_path = os.path.join(save_root, save_name)

    features = features.reshape(features.shape[0], -1)

    np.savez(save_path, feats=features, labels=labels)

    return None, None, None, None


def de_diag(acc, each_angle=False):
    # Exclude identical-view cases
    dividend = acc.shape[1] - 1.
    result = np.sum(acc - np.diag(np.diag(acc)), 1) / dividend
    if not each_angle:
        result = np.mean(result)
    return result


def cross_view_gallery_evaluation(feature, label, seq_type, view, dataset,
                                  metric):
    '''More details can be found: More details can be found in
        [A Comprehensive Study on the Evaluation of Silhouette-based Gait Recognition](https://ieeexplore.ieee.org/document/9928336).
    '''
    probe_seq_dict = {
        'CASIA-B': {
            'NM': ['nm-01'],
            'BG': ['bg-01'],
            'CL': ['cl-01']
        },
        'OUMVLP': {
            'NM': ['00']
        }
    }

    gallery_seq_dict = {
        'CASIA-B': ['nm-02', 'bg-02', 'cl-02'],
        'OUMVLP': ['01']
    }

    msg_mgr = get_msg_mgr()
    acc = {}
    mean_ap = {}
    view_list = sorted(np.unique(view))
    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros(len(view_list)) - 1.
        mean_ap[type_] = np.zeros(len(view_list)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]
            gseq_mask = np.isin(seq_type, gallery_seq_dict[dataset])
            gallery_y = label[gseq_mask]
            gallery_x = feature[gseq_mask, :]
            dist = cuda_dist(probe_x, gallery_x, metric)
            eval_results = compute_ACC_mAP(dist.cpu().numpy(), probe_y,
                                           gallery_y, view[pseq_mask],
                                           view[gseq_mask])
            acc[type_][v1] = np.round(eval_results[0] * 100, 2)
            mean_ap[type_][v1] = np.round(eval_results[1] * 100, 2)

    result_dict = {}
    msg_mgr.log_info(
        '===Cross View Gallery Evaluation (Excluded identical-view cases)===')
    out_acc_str = "========= Rank@1 Acc =========\n"
    out_map_str = "============= mAP ============\n"
    for type_ in probe_seq_dict[dataset].keys():
        avg_acc = np.mean(acc[type_])
        avg_map = np.mean(mean_ap[type_])
        result_dict[f'scalar/test_accuracy/{type_}-Rank@1'] = avg_acc
        result_dict[f'scalar/test_accuracy/{type_}-mAP'] = avg_map
        out_acc_str += f"{type_}:\t{acc[type_]}, mean: {avg_acc:.2f}%\n"
        out_map_str += f"{type_}:\t{mean_ap[type_]}, mean: {avg_map:.2f}%\n"
    # msg_mgr.log_info(f'========= Rank@1 Acc =========')
    msg_mgr.log_info(f'{out_acc_str}')
    # msg_mgr.log_info(f'========= mAP =========')
    msg_mgr.log_info(f'{out_map_str}')
    return result_dict


# Modified From https://github.com/AbnerHqC/GaitSet/blob/master/model/utils/evaluator.py


def single_view_gallery_evaluation(feature, label, seq_type, view, dataset,
                                   metric):
    probe_seq_dict = {
        'CASIA-B': {
            'NM': ['nm-05', 'nm-06'],
            'BG': ['bg-01', 'bg-02'],
            'CL': ['cl-01', 'cl-02']
        },
        'OUMVLP': {
            'NM': ['00']
        },
        'CASIA-E': {
            'NM': [
                'H-scene2-nm-1',
                'H-scene2-nm-2',
                'L-scene2-nm-1',
                'L-scene2-nm-2',
                'H-scene3-nm-1',
                'H-scene3-nm-2',
                'L-scene3-nm-1',
                'L-scene3-nm-2',
                'H-scene3_s-nm-1',
                'H-scene3_s-nm-2',
                'L-scene3_s-nm-1',
                'L-scene3_s-nm-2',
            ],
            'BG': [
                'H-scene2-bg-1', 'H-scene2-bg-2', 'L-scene2-bg-1',
                'L-scene2-bg-2', 'H-scene3-bg-1', 'H-scene3-bg-2',
                'L-scene3-bg-1', 'L-scene3-bg-2', 'H-scene3_s-bg-1',
                'H-scene3_s-bg-2', 'L-scene3_s-bg-1', 'L-scene3_s-bg-2'
            ],
            'CL': [
                'H-scene2-cl-1', 'H-scene2-cl-2', 'L-scene2-cl-1',
                'L-scene2-cl-2', 'H-scene3-cl-1', 'H-scene3-cl-2',
                'L-scene3-cl-1', 'L-scene3-cl-2', 'H-scene3_s-cl-1',
                'H-scene3_s-cl-2', 'L-scene3_s-cl-1', 'L-scene3_s-cl-2'
            ]
        },
        'SUSTech1K': {
            'Normal': ['01-nm'],
            'Bag': ['bg'],
            'Clothing': ['cl'],
            'Carrying': ['cr'],
            'Umberalla': ['ub'],
            'Uniform': ['uf'],
            'Occlusion': ['oc'],
            'Night': ['nt'],
            'Overall': ['01', '02', '03', '04']
        }
    }
    gallery_seq_dict = {
        'CASIA-B': ['nm-01', 'nm-02', 'nm-03', 'nm-04'],
        'OUMVLP': ['01'],
        'CASIA-E':
        ['H-scene1-nm-1', 'H-scene1-nm-2', 'L-scene1-nm-1', 'L-scene1-nm-2'],
        'SUSTech1K': ['00-nm'],
    }
    msg_mgr = get_msg_mgr()
    acc = {}
    view_list = sorted(np.unique(view))
    num_rank = 1
    if dataset == 'CASIA-E':
        view_list.remove("270")
    if dataset == 'SUSTech1K':
        num_rank = 5
    view_num = len(view_list)

    for (type_, probe_seq) in probe_seq_dict[dataset].items():
        acc[type_] = np.zeros((view_num, view_num, num_rank)) - 1.
        for (v1, probe_view) in enumerate(view_list):
            pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                view, probe_view)
            pseq_mask = pseq_mask if 'SUSTech1K' not in dataset else np.any(
                np.asarray([
                    np.char.find(seq_type, probe) >= 0 for probe in probe_seq
                ]),
                axis=0) & np.isin(view, probe_view)  # For SUSTech1K only
            probe_x = feature[pseq_mask, :]
            probe_y = label[pseq_mask]

            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type,
                                    gallery_seq_dict[dataset]) & np.isin(
                                        view, [gallery_view])
                gseq_mask = gseq_mask if 'SUSTech1K' not in dataset else np.any(
                    np.asarray([
                        np.char.find(seq_type, gallery) >= 0
                        for gallery in gallery_seq_dict[dataset]
                    ]),
                    axis=0) & np.isin(view,
                                      [gallery_view])  # For SUSTech1K only
                gallery_y = label[gseq_mask]
                gallery_x = feature[gseq_mask, :]
                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
                acc[type_][v1, v2, :] = np.round(
                    np.sum(
                        np.cumsum(
                            np.reshape(probe_y, [-1, 1])
                            == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) *
                    100 / dist.shape[0], 2)

    result_dict = {}
    msg_mgr.log_info('===Rank-1 (Exclude identical-view cases)===')
    out_str = ""
    for rank in range(num_rank):
        out_str = ""
        for type_ in probe_seq_dict[dataset].keys():
            sub_acc = de_diag(acc[type_][:, :, rank], each_angle=True)
            if rank == 0:
                msg_mgr.log_info(f'{type_}@R{rank+1}: {sub_acc}')
                result_dict[
                    f'scalar/test_accuracy/{type_}@R{rank+1}'] = np.mean(
                        sub_acc)
            out_str += f"{type_}@R{rank+1}: {np.mean(sub_acc):.2f}%\t"
        msg_mgr.log_info(out_str)
    return result_dict


def evaluate_indoor_dataset(data,
                            dataset,
                            metric='euc',
                            cross_view_gallery=False):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data[
        'types'], data['views']
    label = np.array(label)
    view = np.array(view)

    if dataset not in ('CASIA-B', 'OUMVLP', 'CASIA-E', 'SUSTech1K'):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    if cross_view_gallery:
        return cross_view_gallery_evaluation(feature, label, seq_type, view,
                                             dataset, metric)
    else:
        return single_view_gallery_evaluation(feature, label, seq_type, view,
                                              dataset, metric)


def evaluate_all_analysis(data,
                 dataset,
                 metric='euc',
                 dataset_partition='./datasets/Gait3D/Gait3D.json',
                 save_json='show_result/Gait3D.json',
                 enable_save_csv=False,
                 split_eva=False,
                 tb_name='yh_five'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    features = features[:, :, :1]
    print(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-5]
    os.makedirs(save_json_root, exist_ok=True)

    result_dict = {}

    all_save_json = os.path.join(save_json_root, "all.json")

    probe_mask = []
    rel_paths = []
    for id, ty, sq in zip(labels, types, time_seqs):
        rel_paths.append('/'.join([id, ty, sq]))
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)

    gallery_mask = ~np.array(probe_mask)

    probe_features = features[probe_mask]
    gallery_features = features[gallery_mask]

    probe_rel_paths = np.asarray(rel_paths)[probe_mask]
    gallery_rel_paths = np.asarray(rel_paths)[gallery_mask]

    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[gallery_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    try:
        cmc, all_AP, all_INP = evaluate_rank_all_analysis(dist, probe_lbls, gallery_lbls,
                                                          probe_rel_paths, gallery_rel_paths, msg_mgr)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        max_len = min(50, len(cmc))
        for i, r in [1, 5, 10, max_len]:
            results['Rank-{}'.format(50 if i==3 else r)] = cmc[r - 1] * 100
        results['mAP'] = mAP * 100
        results['mINP'] = mINP * 100
        results['gallery_num'] = gallery_features.shape[0]
    except:
        for r in [1, 5, 10, 50]:
            results['Rank-{}'.format(r)] = 0
        results['mAP'] = 0
        results['mINP'] = 0
        results['gallery_num'] = 0

    result_dict['all'] = results

    table_list = ['type'] + list(result_dict['all'].keys())

    table = PrettyTable(table_list)
    table.hrules = ALL
    for k, v in result_dict.items():
        data_list = [k]
        for v_k, v_v in v.items():
            data_list.append(round(v_v, 3))
        print(table_list)
        print(data_list)
        table.add_row(data_list)
    msg_mgr.log_info(table)

    if enable_save_csv:
        # 保存成csv，方便记录
        csv_columns = ['all']
        index_columns = result_dict['all'].keys()
        csv_data = {}
        for data_type in csv_columns:
            if data_type == '0query':
                continue
            for index in index_columns:
                csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                    index]

        filename = 'csv/record_test_all.csv'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_data)

        print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result





def evaluate_yihua_jianzhi(data,
                   dataset,
                   metric='euc',
                   dataset_partition='./datasets/Gait3D/Gait3D.json',
                   save_json='show_result/Gait3D.json',
                   enable_save_csv=True,
                   split_eva=True,
                   tb_name='yh_five',
                   *args,
                   **kwargs):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    features = features[:, :, :1]
    print(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-5]
    os.makedirs(save_json_root, exist_ok=True)

    # eva_type_list = []

    cloth_eva_dict = {}
    scene_eva_dict = {}
    dist_eva_dict = {}
    inout_eva_dict = {}
    sex_eva_dict = {}
    time_eva_dict = {}
    dataset_eva_dict = {}

    for i in range(len(types)):
        if 'noise' in labels[i] or 'query' in types[i]:
            continue
        # cloth_type, scene_type, dist_type, inout_type = types[i].split('_', 3)
        if len(types[i].split('_'))==7:
            cloth_type, scene_type, dist_type, inout_type, sex_type, time_type, dataset_type = types[i].split('_')
        elif len(types[i].split('_'))==8:
            cloth_type, scene_type, scene_type2, dist_type, inout_type, sex_type, time_type, dataset_type = types[i].split('_')
            scene_type = scene_type + '_' + scene_type2
        else:
            raise NotImplementedError("Not Implemented need check dataset")
        cloth_eva_dict.setdefault(cloth_type, []).append(types[i])
        # scene_eva_dict.setdefault(scene_type, []).append(types[i])  # commit...for comp
        dist_eva_dict.setdefault(dist_type, []).append(types[i])
        inout_eva_dict.setdefault(inout_type, []).append(types[i])
        sex_eva_dict.setdefault(sex_type, []).append(types[i])
        time_eva_dict.setdefault(time_type, []).append(types[i])
        dataset_eva_dict.setdefault(dataset_type, []).append(types[i])

    eva_type_dict = {**cloth_eva_dict, **scene_eva_dict, **dist_eva_dict, **inout_eva_dict, **sex_eva_dict,
                     **time_eva_dict, **dataset_eva_dict}
    eva_type_list = list(eva_type_dict.keys())

    result_dict = {}

    if split_eva:
        for eva_type in eva_type_list:
            if eva_type == '0query':
                continue
            save_json = os.path.join(save_json_root, f"{eva_type}.json")
            result = evaluate_type(eva_type_dict[eva_type], data,
                                   probe_sets, msg_mgr, metric, save_json)
            result_dict[eva_type] = result
    all_save_json = os.path.join(save_json_root, "all.json")
    result = evaluate_type('all', data, probe_sets, msg_mgr, metric,
                           all_save_json)
    result_dict['all'] = result

    table_list = ['type'] + list(result_dict['all'].keys())

    table = PrettyTable(table_list)
    table.hrules = ALL
    for k, v in result_dict.items():
        data_list = [k]
        for v_k, v_v in v.items():
            data_list.append(round(v_v, 3))
        print(table_list)
        print(data_list)
        table.add_row(data_list)
    msg_mgr.log_info(f'\n {table}')

    if enable_save_csv:
        # 保存成csv，方便记录
        csv_columns = ['all'] + eva_type_list
        index_columns = result_dict['all'].keys()
        csv_data = {}
        for data_type in csv_columns:
            if data_type == '0query':
                continue
            for index in index_columns:
                csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                    index]

        filename = 'csv/yh_jianzhi_record_test.csv'

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_data)

        print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result


def evaluate_real_scene(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()
    feature, label, seq_type = data['embeddings'], data['labels'], data[
        'types']
    label = np.array(label)

    gallery_seq_type = {
        '0001-1000': ['1', '2'],
        "HID2021": ['0'],
        '0001-1000-test': ['0'],
        'GREW': ['01'],
        'TTG-200': ['1']
    }
    probe_seq_type = {
        '0001-1000': ['3', '4', '5', '6'],
        "HID2021": ['1'],
        '0001-1000-test': ['1'],
        'GREW': ['02'],
        'TTG-200': ['2', '3', '4', '5', '6']
    }

    num_rank = 20
    acc = np.zeros([num_rank]) - 1.
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = label[pseq_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()
    acc = np.round(
        np.sum(
            np.cumsum(
                np.reshape(probe_y, [-1, 1]) == gallery_y[idx[:, 0:num_rank]],
                1) > 0, 0) * 100 / dist.shape[0], 2)

    msg_mgr.log_info('==Rank-1==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[0])))
    msg_mgr.log_info('==Rank-5==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[4])))
    msg_mgr.log_info('==Rank-10==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[9])))
    msg_mgr.log_info('==Rank-20==')
    msg_mgr.log_info('%.3f' % (np.mean(acc[19])))
    return {
        "scalar/test_accuracy/Rank-1": np.mean(acc[0]),
        "scalar/test_accuracy/Rank-5": np.mean(acc[4])
    }


def GREW_submission(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating GREW")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data[
        'types'], data['views']
    label = np.array(label)
    view = np.array(view)
    gallery_seq_type = {'GREW': ['01', '02']}
    probe_seq_type = {'GREW': ['03']}
    gseq_mask = np.isin(seq_type, gallery_seq_type[dataset])
    gallery_x = feature[gseq_mask, :]
    gallery_y = label[gseq_mask]
    pseq_mask = np.isin(seq_type, probe_seq_type[dataset])
    probe_x = feature[pseq_mask, :]
    probe_y = view[pseq_mask]

    num_rank = 20
    dist = cuda_dist(probe_x, gallery_x, metric)
    idx = dist.topk(num_rank, largest=False)[1].cpu().numpy()

    save_path = os.path.join("GREW_result/" +
                             strftime('%Y-%m%d-%H%M%S', localtime()) + ".csv")
    mkdir("GREW_result")
    with open(save_path, "w") as f:
        f.write(
            "videoId,rank1,rank2,rank3,rank4,rank5,rank6,rank7,rank8,rank9,rank10,rank11,rank12,rank13,rank14,rank15,rank16,rank17,rank18,rank19,rank20\n"
        )
        for i in range(len(idx)):
            r_format = [int(idx) for idx in gallery_y[idx[i, 0:num_rank]]]
            output_row = '{}' + ',{}' * num_rank + '\n'
            f.write(output_row.format(probe_y[i], *r_format))
        print("GREW result saved to {}/{}".format(os.getcwd(), save_path))
    return


def HID_submission(data, dataset, rerank=True, metric='euc'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info("Evaluating HID")
    feature, label, seq_type = data['embeddings'], data['labels'], data[
        'views']
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]
    if rerank:
        feat = np.concatenate([probe_x, gallery_x])
        dist = cuda_dist(feat, feat, metric).cpu().numpy()
        msg_mgr.log_info("Starting Re-ranking")
        re_rank = re_ranking(dist,
                             probe_x.shape[0],
                             k1=6,
                             k2=6,
                             lambda_value=0.3)
        idx = np.argsort(re_rank, axis=1)
    else:
        dist = cuda_dist(probe_x, gallery_x, metric)
        idx = dist.cpu().sort(1)[1].numpy()

    save_path = os.path.join("HID_result/" +
                             strftime('%Y-%m%d-%H%M%S', localtime()) + ".csv")
    mkdir("HID_result")
    with open(save_path, "w") as f:
        f.write("videoID,label\n")
        for i in range(len(idx)):
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))
        print("HID result saved to {}/{}".format(os.getcwd(), save_path))
    return


def evaluate_segmentation(data, dataset):
    labels = data['mask']
    pred = data['pred']
    miou = mean_iou(pred, labels)
    get_msg_mgr().log_info('mIOU: %.3f' % (miou.mean()))
    return {"scalar/test_accuracy/mIOU": miou}


def write_results(dist,
                  probe_mask,
                  labels,
                  cams,
                  time_seqs,
                  save_json,
                  gallery_mask=None):
    """

    Args:
        dist:
        probe_mask:
        labels:
        cams:
        time_seqs:
        save_json:

    Returns:

    """
    seq_path_list = []
    for label, cam, time_seq in zip(labels, cams, time_seqs):
        seq_path = os.path.join(label, cam, time_seq)
        seq_path_list.append(seq_path)
    seq_path_list = np.array(seq_path_list)
    probe_seq_path_list = seq_path_list[probe_mask]
    labels = np.array(labels)
    probe_label = labels[probe_mask]
    if gallery_mask is None:
        gallery_seq_path_list = seq_path_list[~probe_mask]
        gallery_label = labels[~probe_mask]
    else:
        gallery_seq_path_list = seq_path_list[gallery_mask]
        gallery_label = labels[gallery_mask]
    result = {}

    for i, probe_dist in enumerate(dist):
        probe_path = probe_seq_path_list[i]
        result[probe_path] = []
        probe_result = result[probe_path]
        sorted_dist_index = np.argsort(probe_dist)
        for j, sort_index in enumerate(sorted_dist_index):
            gallery_path = gallery_seq_path_list[sort_index]

            if gallery_label[sort_index] == probe_label[i]:
                # 加入全部列表，并把正例标记成 1
                probe_result.append(
                    [gallery_path,
                     str(dist[i][sort_index]),
                     str(1)])
            else:
                # 加入全部列表，并把负例标记为 0
                probe_result.append(
                    [gallery_path,
                     str(dist[i][sort_index]),
                     str(0)])

    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, 'w') as f:
        json.dump(result, f, indent=4)

    return result


def evaluate_Gait3D(data,
                    dataset,
                    metric='euc',
                    dataset_partition='./datasets/Gait3D/Gait3D.json', split_eva=None):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, cams, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    import json
    # path = './datasets/Gait3D/Gait3D.json'
    # path = 'E:/dataset/Gait3D/YH_LITE-silhs/Gait3D.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_clean.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_train_2.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_train_2_2.json'
    # print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']

    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    np.save("./probe_features.npy", probe_features)
    np.save("./gallery_features.npy", gallery_features)
    dist_all = cuda_dist_all(probe_features, gallery_features, metric)

    # 保存dist
    # np.save("./dist.npy", dist)
    # dist = np.load("./dist.npy")
    def print_rank_map(dist,
                       probe_lbls,
                       gallery_lbls,
                       results,
                       metric='official'):
        cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        tmp_results = {}  # just for print
        sep = '_' + metric if metric != 'official' else ''
        for r in [1, 5, 10]:
            tmp_results['{}/Rank-{}'.format(sep, r)] = cmc[r - 1] * 100
        tmp_results[f"{sep}/mAP"] = mAP * 100
        tmp_results[f"{sep}/mINP"] = mINP * 100

        tabel = PrettyTable(['Rank-1', 'Rank-5', 'Rank-10', 'mAP', 'mINP'])
        tabel.hrules = ALL

        tabel.add_row([
            round(tmp_results[f'{sep}/Rank-1'], 3),
            round(tmp_results[f'{sep}/Rank-5'], 3),
            round(tmp_results[f'{sep}/Rank-10'], 3),
            round(tmp_results[f'{sep}/mAP'], 3),
            round(tmp_results[f'{sep}/mINP'], 3)
        ])

        print(tabel)

        # # 分隔符
        results.update(tmp_results)
        # sep = f"{'-'*8}{metric}{'-'*8}"
        # msg_mgr.log_info(sep)
        # msg_mgr.log_info(tmp_results)
        # msg_mgr.log_info(f"{'-' * len(sep)}")
        # return

    for k, v in dist_all.items():
        print_rank_map(v.cpu().numpy(), probe_lbls, gallery_lbls, results, k)
    # print(results)

    return results


def evaluate_CCGR(data, dataset, metric='euc'):
    get_msg_mgr().log_info("Evaluating CCGR")
    feature, label, seq_type, view = data['embeddings'], data['labels'], data[
        'types'], data['views']
    return feature, label, seq_type, view


def evaluate_Gait3D_old(data,
                        dataset,
                        metric='euc',
                        dataset_partition='./datasets/Gait3D/Gait3D.json',
                        save_json='show_result/Gait3D.json'):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, cams, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']

    # path = './datasets/Gait3D/Gait3D.json'
    # path = 'E:/dataset/Gait3D/YH_LITE-silhs/Gait3D.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_clean.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_train_2.json'
    # path = r'.\datasets\Gait3D\Gait3D2024-03-12_train_2_2.json'
    # print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']

    probe_mask = []
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
    probe_mask = np.array(probe_mask)

    # probe_features = features[:probe_num]
    probe_features = features[probe_mask]
    # gallery_features = features[probe_num:]
    gallery_features = features[~probe_mask]
    # probe_lbls = np.asarray(labels[:probe_num])
    # gallery_lbls = np.asarray(labels[probe_num:])
    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[~probe_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()

    cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    for r in [1, 5, 10]:
        results['Rank-{}'.format(r)] = cmc[r - 1] * 100
    results['mAP'] = mAP * 100
    results['mINP'] = mINP * 100

    table = PrettyTable(
        ['Rank-1', 'Rank-5', 'Rank-10', 'mAP', 'mINP', 'gallery_num'])
    table.hrules = ALL
    table.add_row([
        round(results['Rank-1'], 3),
        round(results['Rank-5'], 3),
        round(results['Rank-10'], 3),
        round(results['mAP'], 3),
        round(results['mINP'], 3), gallery_features.shape[0]
    ])

    print(table)

    # print_csv_format(dataset_name, results)
    # msg_mgr.log_info(results)

    write_results(dist, probe_mask, labels, cams, time_seqs, save_json)

    csv_filename = 'csv/Gait3D_record_old_test.csv'
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    with open(csv_filename, 'a', newline='') as file:
        writer = csv.DictWriter(
            file, fieldnames=['Rank-1', 'Rank-5', 'Rank-10', 'mAP', 'mINP'])
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(results)

    return results






def evaluate_type(eva_type, data, probe_sets, msg_mgr, metric, save_json):
    features, labels, cams, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    # features = features[:, :, ::4]
    print("in line 717 evaluator.py need ca")
    probe_mask = []
    gallery_mask = []
    # print(probe_sets)
    for id, ty, sq in zip(labels, cams, time_seqs):
        if '-'.join([id, ty, sq]) in probe_sets:
            probe_mask.append(True)
        else:
            probe_mask.append(False)
        if eva_type == 'all':
            continue
        if isinstance(eva_type, list):
            if ty in eva_type or ('noise' in id):
                gallery_mask.append(True)
            else:
                gallery_mask.append(False)
        else:
            if ty == eva_type or ('noise' in id):
                gallery_mask.append(True)
            else:
                gallery_mask.append(False)

    if eva_type == 'all':
        gallery_mask = ~np.array(probe_mask)

    probe_features = features[probe_mask]
    gallery_features = features[gallery_mask]

    probe_lbls = np.asarray(labels)[probe_mask]
    gallery_lbls = np.asarray(labels)[gallery_mask]

    results = {}
    msg_mgr.log_info(f"The test metric you choose is {metric}.")
    dist = cuda_dist(probe_features, gallery_features, metric).cpu().numpy()
    # dist = euc_dist(probe_features, gallery_features)
    # 拍平并保存
    np.save("./dist.npy", dist)
    try:
        cmc, all_AP, all_INP = evaluate_rank(dist, probe_lbls, gallery_lbls)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        max_len = min(50, len(cmc))
        for i, r in enumerate([1, 5, 10, max_len]):
            results['Rank-{}'.format(50 if i==3 else r)] = cmc[r - 1] * 100
        results['mAP'] = mAP * 100
        results['mINP'] = mINP * 100
        results['gallery_num'] = gallery_features.shape[0]
    except:
        for r in [1, 5, 10, 50]:
            results['Rank-{}'.format(r)] = 0
        results['mAP'] = 0
        results['mINP'] = 0
        results['gallery_num'] = 0

    # print_csv_format(dataset_name, results)
    # msg_mgr.log_info(results)

    # write_results(dist,
    #               probe_mask,
    #               labels,
    #               cams,
    #               time_seqs,
    #               save_json,
    #               gallery_mask=gallery_mask)

    return results


def evaluate_yh_five(data,
                     dataset,
                     metric='euc',
                     dataset_partition='./datasets/Gait3D/Gait3D.json',
                     save_json='show_result/Gait3D.json',
                     enable_save_csv=True,
                     split_eva=True,
                     tb_name='yh_five'):
    yh_five_list = [
        '1normal', '2change_cloth', '3no_face', '4no_face_leg', '5no_face_run',
        '6no_face_box'
    ]
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")

    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    # features = features[:, :, :1]
    msg_mgr.log_info(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-4]
    os.makedirs(save_json_root, exist_ok=True)

    eva_type_list = []

    for i in range(len(types)):
        if labels[i] == 'noise' or types[i] == '0query':
            continue
        if types[i] not in eva_type_list:
            eva_type_list.append(types[i])

    eva_type_list = sorted(eva_type_list)

    result_dict = {}

    if split_eva:
        for eva_type in eva_type_list:
            if eva_type == '0query':
                continue
            save_json = os.path.join(save_json_root, f"{eva_type}.json")
            result = evaluate_type(eva_type, data, probe_sets, msg_mgr, metric,
                                   save_json)
            result_dict[eva_type] = result
    all_save_json = os.path.join(save_json_root, "all.json")
    result = evaluate_type('all', data, probe_sets, msg_mgr, metric,
                           all_save_json)
    result_dict['all'] = result

    table_list = ['type'] + list(result_dict['all'].keys())

    table = PrettyTable(table_list)
    table.hrules = ALL
    for k, v in result_dict.items():
        data_list = [k]
        for v_k, v_v in v.items():
            data_list.append(round(v_v, 3))
        print(table_list)
        print(data_list)
        table.add_row(data_list)
    msg_mgr.log_info(table)

    if enable_save_csv:
        # 保存成csv，方便记录
        csv_columns = ['all'] + eva_type_list
        index_columns = result_dict['all'].keys()
        csv_data = {}
        for data_type in csv_columns:
            if data_type == '0query':
                continue
            for index in index_columns:
                csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                    index]

        if eva_type_list == yh_five_list:
            filename = 'csv/yh_five_record_test.csv'
        else:
            filename = f'csv/{"_".join(eva_type_list[:4])}'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_data)

        print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result


def evaluate_yh_40(data,
                   dataset,
                   metric='euc',
                   dataset_partition='./datasets/Gait3D/Gait3D.json',
                   save_json='show_result/Gait3D.json',
                   enable_save_csv=False,
                   split_eva=True,
                   tb_name='yh_five',
                   *args,
                   **kwargs):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    # features = features[:, :, :1]
    print(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-4]
    os.makedirs(save_json_root, exist_ok=True)

    # eva_type_list = []

    cloth_eva_dict = {}
    sec_eva_dict = {}
    day_night_eva_dict = {}

    for i in range(len(types)):
        if labels[i] == 'noise' or types[i] == '0query':
            continue
        tmp_sec = types[i]
        tmp_sec_sp = tmp_sec.split('_')
        cloth_type = tmp_sec_sp[0]
        night_type_str = tmp_sec_sp[1]
        if 'night' in night_type_str:
            night_type = 'night'
            sec_type = '_'.join(tmp_sec_sp[2:])
        else:
            night_type = 'day'
            start_index = 0
            for sec_i, s in enumerate(night_type_str):
                if not s.isdigit():
                    start_index = sec_i
                    break
            sec_first_type = night_type_str[start_index:]
            sec_type = '_'.join([sec_first_type] + tmp_sec_sp[2:])
            if sec_type.isspace():
                sec_type = 'normal'

        cloth_eva_dict.setdefault(cloth_type, []).append(types[i])
        day_night_eva_dict.setdefault(night_type, []).append(types[i])
        sec_eva_dict.setdefault(sec_type, []).append(types[i])

    for eva_type_dict in [cloth_eva_dict, day_night_eva_dict, sec_eva_dict]:

        eva_type_list = sorted(list(eva_type_dict.keys()))

        result_dict = {}

        if split_eva:
            for eva_type in eva_type_list:
                if eva_type == '0query':
                    continue
                save_json = os.path.join(save_json_root, f"{eva_type}.json")
                result = evaluate_type(eva_type_dict[eva_type], data,
                                       probe_sets, msg_mgr, metric, save_json)
                result_dict[eva_type] = result
        all_save_json = os.path.join(save_json_root, "all.json")
        result = evaluate_type('all', data, probe_sets, msg_mgr, metric,
                               all_save_json)
        result_dict['all'] = result

        table_list = ['type'] + list(result_dict['all'].keys())

        table = PrettyTable(table_list)
        table.hrules = ALL
        for k, v in result_dict.items():
            data_list = [k]
            for v_k, v_v in v.items():
                data_list.append(round(v_v, 3))
            print(table_list)
            print(data_list)
            table.add_row(data_list)
        msg_mgr.log_info(f'\n {table}')

        if enable_save_csv:
            # 保存成csv，方便记录
            csv_columns = ['all'] + eva_type_list
            index_columns = result_dict['all'].keys()
            csv_data = {}
            for data_type in csv_columns:
                if data_type == '0query':
                    continue
                for index in index_columns:
                    csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                        index]

            filename = 'csv/yh_40_record_test.csv'

            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, 'a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=csv_data.keys())
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(csv_data)

            print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result



def evaluate_yh_all(data,
                   dataset,
                   metric='euc',
                   dataset_partition='./datasets/Gait3D/Gait3D.json',
                   save_json='show_result/Gait3D.json',
                   enable_save_csv=True,
                   split_eva=True,
                   tb_name='yh_five',
                   *args,
                   **kwargs):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    features = features[:, :, :1]
    print(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-4]
    os.makedirs(save_json_root, exist_ok=True)

    # eva_type_list = []

    cloth_eva_dict = {}
    sec_eva_dict = {}
    day_night_eva_dict = {}
    dh_yh_eva_dict = {}

    for i in range(len(types)):
        if labels[i] == 'noise' or types[i] == '0query':
            continue
        tmp_sec = types[i]
        tmp_sec_sp = tmp_sec.split('_')
        cloth_type = tmp_sec_sp[0]
        night_type_str = tmp_sec_sp[1]
        dh_yh_eva = tmp_sec_sp[-1]
        if 'night' in night_type_str:
            night_type = 'night'
            sec_type = '_'.join(tmp_sec_sp[2:])
        else:
            night_type = 'day'
            start_index = 0
            for sec_i, s in enumerate(night_type_str):
                if not s.isdigit():
                    start_index = sec_i
                    break
            sec_first_type = night_type_str[start_index:]
            sec_type = '_'.join([sec_first_type] + tmp_sec_sp[2:])
            if sec_type.isspace():
                sec_type = 'normal'
            sec_type_sp = sec_type.split('_')
            sec_new_list = []
            for sec in sec_type_sp:
                if sec in ['yh', 'dh']:
                    continue
                sec_new_list.append(sec)
            sec_type = '_'.join(sec_new_list)

        cloth_eva_dict.setdefault(cloth_type, []).append(types[i])
        day_night_eva_dict.setdefault(night_type, []).append(types[i])
        sec_eva_dict.setdefault(sec_type, []).append(types[i])
        dh_yh_eva_dict.setdefault(dh_yh_eva, []).append(types[i])
    result_dict = {}
    sec_list = []
    for eva_type_dict in [dh_yh_eva_dict, cloth_eva_dict, day_night_eva_dict, sec_eva_dict]:

        eva_type_list = sorted(list(eva_type_dict.keys()))
        sec_list.extend(eva_type_list)

        if split_eva:
            for eva_type in eva_type_list:
                if eva_type == '0query':
                    continue
                save_json = os.path.join(save_json_root, f"{eva_type}.json")
                result = evaluate_type(eva_type_dict[eva_type], data,
                                       probe_sets, msg_mgr, metric, save_json)
                result_dict[eva_type] = result

    all_save_json = os.path.join(save_json_root, "all.json")
    result = evaluate_type('all', data, probe_sets, msg_mgr, metric,
                           all_save_json)
    result_dict['all'] = result
    sec_list.append('all')

    table_list = ['type'] + list(result_dict['all'].keys())

    table = PrettyTable(table_list)
    table.hrules = ALL
    # for k, v in result_dict.items():
    for k in sec_list:
        v = result_dict[k]
        data_list = [k]
        for v_k, v_v in v.items():
            data_list.append(round(v_v, 3))
        # print(table_list)
        # print(data_list)
        table.add_row(data_list)
    msg_mgr.log_info(f'\n {table}')

    if enable_save_csv:
        # 保存成csv，方便记录
        csv_columns = ['all'] + eva_type_list
        index_columns = result_dict['all'].keys()
        csv_data = {}
        for data_type in csv_columns:
            if data_type == '0query':
                continue
            for index in index_columns:
                csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                    index]

        filename = 'csv/yh_40_record_test.csv'

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_data)

        print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result

def evaluate_yh_all_old(data,
                   dataset,
                   metric='euc',
                   dataset_partition='./datasets/Gait3D/Gait3D.json',
                   save_json='show_result/Gait3D.json',
                   enable_save_csv=False,
                   split_eva=False,
                   tb_name='yh_five',
                   *args,
                   **kwargs):
    msg_mgr = get_msg_mgr()
    msg_mgr.log_info(f"dataset_partition is {dataset_partition}")
    features, labels, types, time_seqs = data['embeddings'], data[
        'labels'], data['types'], data['views']
    # features = features[:, :, :1]
    print(features.shape)
    print(f"in {__file__},set path is {dataset_partition}")
    probe_sets = json.load(open(dataset_partition, 'rb'))['PROBE_SET']
    save_json_root = save_json[:-4]
    os.makedirs(save_json_root, exist_ok=True)

    # eva_type_list = []

    cloth_eva_dict = {}
    sec_eva_dict = {}
    day_night_eva_dict = {}
    dh_yh_eva_dict = {}

    for i in range(len(types)):
        if labels[i] == 'noise' or types[i] == '0query':
            continue
        tmp_sec = types[i]
        tmp_sec_sp = tmp_sec.split('_')
        cloth_type = tmp_sec_sp[0]
        night_type_str = tmp_sec_sp[1]
        dh_yh_eva = tmp_sec_sp[-1]
        if 'night' in night_type_str:
            night_type = 'night'
            sec_type = '_'.join(tmp_sec_sp[2:])
        else:
            night_type = 'day'
            start_index = 0
            for sec_i, s in enumerate(night_type_str):
                if not s.isdigit():
                    start_index = sec_i
                    break
            sec_first_type = night_type_str[start_index:]
            sec_type = '_'.join([sec_first_type] + tmp_sec_sp[2:])
            if sec_type.isspace():
                sec_type = 'normal'
            sec_type_sp = sec_type.split('_')
            sec_new_list = []
            for sec in sec_type_sp:
                if sec in ['yh', 'dh']:
                    continue
                sec_new_list.append(sec)
            sec_type = '_'.join(sec_new_list)

        cloth_eva_dict.setdefault(cloth_type, []).append(types[i])
        day_night_eva_dict.setdefault(night_type, []).append(types[i])
        sec_eva_dict.setdefault(sec_type, []).append(types[i])
        dh_yh_eva_dict.setdefault(dh_yh_eva, []).append(types[i])
    result_dict = {}
    sec_list = []
    for eva_type_dict in [dh_yh_eva_dict, cloth_eva_dict, day_night_eva_dict, sec_eva_dict]:

        eva_type_list = sorted(list(eva_type_dict.keys()))
        sec_list.extend(eva_type_list)

        if split_eva:
            for eva_type in eva_type_list:
                if eva_type == '0query':
                    continue
                save_json = os.path.join(save_json_root, f"{eva_type}.json")
                result = evaluate_type(eva_type_dict[eva_type], data,
                                       probe_sets, msg_mgr, metric, save_json)
                result_dict[eva_type] = result

    all_save_json = os.path.join(save_json_root, "all.json")
    result = evaluate_type('all', data, probe_sets, msg_mgr, metric,
                           all_save_json)
    result_dict['all'] = result
    sec_list.append('all')

    table_list = ['type'] + list(result_dict['all'].keys())

    table = PrettyTable(table_list)
    table.hrules = ALL
    # for k, v in result_dict.items():
    for k in sec_list:
        v = result_dict[k]
        data_list = [k]
        for v_k, v_v in v.items():
            data_list.append(round(v_v, 3))
        # print(table_list)
        # print(data_list)
        table.add_row(data_list)
    msg_mgr.log_info(f'\n {table}')

    if enable_save_csv:
        # 保存成csv，方便记录
        csv_columns = ['all'] + eva_type_list
        index_columns = result_dict['all'].keys()
        csv_data = {}
        for data_type in csv_columns:
            if data_type == '0query':
                continue
            for index in index_columns:
                csv_data[f"{data_type}_{index}"] = result_dict[data_type][
                    index]

        filename = 'csv/yh_40_record_test.csv'

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=csv_data.keys())
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(csv_data)

        print(f"新数据已追加到 {filename}")

    return_result = {}
    for key, value in result_dict['all'].items():
        if key == 'gallery_num':
            continue
        return_result[f'{tb_name}/{key}'] = value

    return return_result



def evaluate_CCPG(data, dataset, metric='euc'):
    msg_mgr = get_msg_mgr()

    feature, label, seq_type, view = data['embeddings'], data['labels'], data[
        'types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()

    view_num = len(view_list)

    probe_seq_dict = {
        'CCPG': [["U0_D0_BG", "U0_D0"], ["U3_D3"], ["U1_D0"], ["U0_D0_BG"]]
    }

    gallery_seq_dict = {
        'CCPG': [["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"], ["U0_D0"]]
    }
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)
    num_rank = 5
    acc = np.zeros(
        [len(probe_seq_dict[dataset]), view_num, view_num, num_rank]) - 1.

    ap_save = []
    cmc_save = []
    cmc5_save = []
    cmc10_save = []
    minp = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)
        gallery_x = feature[gseq_mask, :]
        # print("gallery_x", gallery_x.shape)
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]

        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        msg_mgr.log_info(
            ("gallery length", len(gallery_y), gallery_seq, "probe length",
             len(probe_y), probe_seq))
        distmat = cuda_dist(probe_x, gallery_x, metric).cpu().numpy()
        # cmc, ap = evaluate(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc, ap, inp = evaluate_many(distmat, probe_y, gallery_y, probe_view,
                                     gallery_view)
        ap_save.append(ap)
        cmc_save.append(cmc[0])
        cmc5_save.append(cmc[4])
        cmc10_save.append(cmc[9])
        minp.append(inp)

    # print(ap_save, cmc_save)

    msg_mgr.log_info(
        '===Rank-1 (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (cmc_save[0] * 100, cmc_save[1] * 100, cmc_save[2] * 100,
                      cmc_save[3] * 100))
    msg_mgr.log_info(
        '===Rank-5 (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (cmc5_save[0] * 100, cmc5_save[1] * 100,
                      cmc5_save[2] * 100, cmc5_save[3] * 100))
    msg_mgr.log_info(
        '===Rank-10 (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (cmc10_save[0] * 100, cmc10_save[1] * 100,
                      cmc10_save[2] * 100, cmc10_save[3] * 100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (ap_save[0] * 100, ap_save[1] * 100, ap_save[2] * 100,
                      ap_save[3] * 100))

    msg_mgr.log_info(
        '===mAP (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                     (ap_save[0] * 100, ap_save[1] * 100, ap_save[2] * 100,
                      ap_save[3] * 100))

    msg_mgr.log_info(
        '===mINP (Exclude identical-view cases for Person Re-Identification)==='
    )
    msg_mgr.log_info(
        'CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
        (minp[0] * 100, minp[1] * 100, minp[2] * 100, minp[3] * 100))

    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        # for gallery_seq in gallery_seq_dict[dataset]:
        gallery_seq = gallery_seq_dict[dataset][p]
        for (v1, probe_view) in enumerate(view_list):
            for (v2, gallery_view) in enumerate(view_list):
                gseq_mask = np.isin(seq_type, gallery_seq) & np.isin(
                    view, [gallery_view])
                gallery_x = feature[gseq_mask, :]
                gallery_y = label[gseq_mask]

                pseq_mask = np.isin(seq_type, probe_seq) & np.isin(
                    view, [probe_view])
                probe_x = feature[pseq_mask, :]
                probe_y = label[pseq_mask]

                dist = cuda_dist(probe_x, gallery_x, metric)
                idx = dist.sort(1)[1].cpu().numpy()
                # print(p, v1, v2, "\n")
                acc[p, v1, v2, :] = np.round(
                    np.sum(
                        np.cumsum(
                            np.reshape(probe_y, [-1, 1])
                            == gallery_y[idx[:, 0:num_rank]], 1) > 0, 0) *
                    100 / dist.shape[0], 2)
    result_dict = {}
    for i in range(1):
        msg_mgr.log_info('===Rank-%d (Include identical-view cases)===' %
                         (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                         (np.mean(acc[0, :, :, i]), np.mean(acc[1, :, :, i]),
                          np.mean(acc[2, :, :, i]), np.mean(acc[3, :, :, i])))
    for i in range(1):
        msg_mgr.log_info('===Rank-%d (Exclude identical-view cases)===' %
                         (i + 1))
        msg_mgr.log_info('CL: %.3f,\tUP: %.3f,\tDN: %.3f,\tBG: %.3f' %
                         (de_diag(acc[0, :, :, i]), de_diag(acc[1, :, :, i]),
                          de_diag(acc[2, :, :, i]), de_diag(acc[3, :, :, i])))
    result_dict["scalar/test_accuracy/CL"] = acc[0, :, :, i]
    result_dict["scalar/test_accuracy/UP"] = acc[1, :, :, i]
    result_dict["scalar/test_accuracy/DN"] = acc[2, :, :, i]
    result_dict["scalar/test_accuracy/BG"] = acc[3, :, :, i]
    np.set_printoptions(precision=2, floatmode='fixed')
    for i in range(1):
        msg_mgr.log_info(
            '===Rank-%d of each angle (Exclude identical-view cases)===' %
            (i + 1))
        msg_mgr.log_info('CL: {}'.format(de_diag(acc[0, :, :, i], True)))
        msg_mgr.log_info('UP: {}'.format(de_diag(acc[1, :, :, i], True)))
        msg_mgr.log_info('DN: {}'.format(de_diag(acc[2, :, :, i], True)))
        msg_mgr.log_info('BG: {}'.format(de_diag(acc[3, :, :, i], True)))
    return result_dict
