import numpy as np
import torch

# AUC
from sklearn.metrics import roc_auc_score

# ACC
from sklearn.metrics import accuracy_score, precision_score

# clDice
from skimage.morphology import skeletonize

# Hausdorff
from scipy.spatial.distance import directed_hausdorff, cdist

# Contiuity
from skimage.morphology import label as sklabel

from monai.metrics.cumulative_average import CumulativeAverage



def cal_Dice(pred: torch.Tensor, label: torch.Tensor):
    tmp = pred + label
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pred)
    c = np.sum(label)

    dice = (2 * a) / (b + c)
    return dice


def cal_Dice_temp(pred: torch.Tensor, label: torch.Tensor):
    return temporal_warpper_metrics(eval_func=cal_Dice, pred=pred, label=label)


def cal_cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v * s) / np.sum(s)


def cal_clDice(pred: torch.Tensor, label: torch.Tensor):
    """[this function computes the cldice metric]

    Args:
        v_p ([bool]): [predicted image]
        v_l ([bool]): [ground truth image]

    Returns:
        [float]: [cldice metric]
    """
    tprec = cal_cl_score(pred, skeletonize(label))
    tsens = cal_cl_score(label, skeletonize(pred))
    return 2 * tprec * tsens / (tprec + tsens)


def cal_clDice_temp(pred: torch.Tensor, label: torch.Tensor):
    return temporal_warpper_metrics(eval_func=cal_clDice, pred=pred, label=label)


def cal_auc(logit: torch.Tensor, label: torch.Tensor):
    logit = logit[:, 0, :, :, :]
    auc = roc_auc_score(
        label.cpu().ravel(),
        logit.cpu().ravel(),
    )

    return auc


def cal_auc_temp(pred: torch.Tensor, label: torch.Tensor):
    return temporal_warpper_metrics(eval_func=cal_auc, pred=pred, label=label)


def cal_hausdorff(pred: torch.Tensor, label: torch.Tensor):
    pred_skel = skeletonize(pred)
    label_skel = skeletonize(label)

    pred_skel = pred_skel * label

    pred_pc = np.array(np.where(pred_skel > 0)).T
    label_pc = np.array(np.where(label_skel > 0)).T

    dist_matrix = cdist(pred_pc, label_pc)
    dist_pred2label = np.min(dist_matrix, axis=0)
    dist_label2pred = np.min(dist_matrix, axis=1)
    dist_pred2label = np.sort(dist_pred2label)
    dist_label2pred = np.sort(dist_label2pred)

    hausdorff_pred2label = np.percentile(dist_pred2label, 95)
    hausdorff_label2pred = np.percentile(dist_label2pred, 95)

    return min(hausdorff_pred2label, hausdorff_label2pred)


def cal_hausdorff_temp(pred: torch.Tensor, label: torch.Tensor):
    return temporal_warpper_metrics(eval_func=cal_hausdorff, pred=pred, label=label)


def cal_accuracy(pred: torch.Tensor, label: torch.Tensor):
    pred_ = pred.detach().cpu().numpy()
    label_ = label.detach().cpu().numpy()

    acc = accuracy_score(
        label_.ravel(),
        pred_.ravel(),
    )

    return acc


def cal_continuity(pred: torch.Tensor, label: torch.Tensor):
    pred_ = pred.detach().cpu().numpy()
    label_ = label.detach().cpu().numpy()

    pred_ = np.squeeze(pred_)
    label_ = np.squeeze(label_)

    pred_masked = pred_ * label_

    _, num_cc = sklabel(pred_masked, return_num=True)

    flag = float(num_cc < 2)

    return flag


def cal_clContinuity(pred: torch.Tensor, label: torch.Tensor):
    pred_ = pred.detach().cpu().numpy()
    label_ = label.detach().cpu().numpy()

    # tprec = cal_cl_score(pred_, skeletonize(label_))
    tsens = cal_cl_score(label_, skeletonize(pred_))
    return tsens

def temporal_warpper_metrics(eval_func, pred: torch.Tensor, label: torch.Tensor):
    pred_ = pred.detach().cpu().numpy().squeeze()
    label_ = label.detach().cpu().numpy().squeeze()

    assert pred_.shape == label_.shape
    if pred_.ndim == 3:
        T, H, W = pred_.shape

        indicator_AM = CumulativeAverage()
        for t in range(T):
            indicator = eval_func(pred_[t], label_[t])
            indicator_AM.append(indicator)
        return indicator_AM.aggregate().item()
    elif pred_.ndim == 2:
        return eval_func(pred_, label_)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logit = torch.randn((256, 256), dtype=torch.float32).to(device)
    pred = torch.where(logit > 0.5, 1, 0)
    label = torch.randn((256, 256), dtype=torch.float32).to(device)
    label = torch.where(label > 0.5, 1, 0)



    hausdorff = cal_hausdorff(pred, label)
    print("hausdorff = {}".format(hausdorff))
