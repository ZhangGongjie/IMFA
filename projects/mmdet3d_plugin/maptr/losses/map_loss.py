import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from torch.nn.functional import l1_loss, mse_loss, smooth_l1_loss

from mmdet.models.builder import LOSSES
from mmdet.models import weighted_loss
from mmdet.core.bbox.match_costs.builder import MATCH_COST

from projects.mmdet3d_plugin.maptr.losses.diff_ras import SoftLane, SoftPolygon


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_dir_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_dir
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            # loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum()
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

@mmcv.jit(derivate=True, coderize=True)
def custom_weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): num_sample, num_order, num_pts, num_coords
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Average factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        raise ValueError('avg_factor should not be none for OrderedPtsL1Loss')
        # loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # import pdb;pdb.set_trace()
            loss = loss.permute(1,0,2,3).contiguous()
            loss = loss.sum((1,2,3))
            loss = loss / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def custom_weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weighted_dir_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    avg_factor=None, **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, avg_factor=2)
    tensor(1.5000)
    """

    @functools.wraps(loss_func)
    def wrapper(pred,
                target,
                weight=None,
                reduction='mean',
                avg_factor=None,
                **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = custom_weight_dir_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_smooth_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss =smooth_l1_loss(pred,target, reduction='none')
    # import pdb;pdb.set_trace()
    return loss

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_loss
def ordered_pts_l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_pts, num_coords]
        target (torch.Tensor): shape [num_samples, num_order, num_pts, num_coords]

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0
    pred = pred.unsqueeze(1).repeat(1, target.size(1),1,1)
    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss

@mmcv.jit(derivate=True, coderize=True)
@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """ Dir cosine similiarity loss
    pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
    target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction='none')
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0,1), target.flatten(0,1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss

@LOSSES.register_module()
class OrderedPtsSmoothL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsSmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_bbox = self.loss_weight * ordered_pts_smooth_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@LOSSES.register_module()
class PtsDirCosLoss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsDirCosLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_dir = self.loss_weight * pts_dir_cos_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_dir



@LOSSES.register_module()
class PtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(PtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_bbox = self.loss_weight * pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox

@LOSSES.register_module()
class OrderedPtsL1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(OrderedPtsL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        
        loss_bbox = self.loss_weight * ordered_pts_l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox




@MATCH_COST.register_module()
class OrderedPtsSmoothL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1).unsqueeze(1).repeat(1,num_gts*num_orders,1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1).unsqueeze(0).repeat(bbox_pred.size(0),1,1)
        
        bbox_cost = smooth_l1_loss(bbox_pred, gt_bboxes, reduction='none').sum(-1)
        
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class PtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.view(num_gts,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y). 
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        bbox_pred = bbox_pred.view(bbox_pred.size(0),-1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts*num_orders,-1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight

@MATCH_COST.register_module()
class MyChamferDistanceCost:
    def __init__(self, loss_src_weight=1., loss_dst_weight=1.):
        # assert mode in ['smooth_l1', 'l1', 'l2']
        # self.mode = mode
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def __call__(self, src, dst,src_weight=1.0,dst_weight=1.0,):
        """
        pred_pts (Tensor): normed coordinate(x,y), shape (num_q, num_pts_M, 2)
        gt_pts (Tensor): normed coordinate(x,y), shape (num_gt, num_pts_N, 2)
        """
        # criterion_mode = self.mode
        # if criterion_mode == 'smooth_l1':
        #     criterion = smooth_l1_loss
        # elif criterion_mode == 'l1':
        #     criterion = l1_loss
        # elif criterion_mode == 'l2':
        #     criterion = mse_loss
        # else:
        #     raise NotImplementedError
        # import pdb;pdb.set_trace()
        src_expand = src.unsqueeze(1).repeat(1,dst.shape[0],1,1)
        dst_expand = dst.unsqueeze(0).repeat(src.shape[0],1,1,1)
        # src_expand = src.unsqueeze(2).unsqueeze(1).repeat(1,dst.shape[0], 1, dst.shape[1], 1)
        # dst_expand = dst.unsqueeze(1).unsqueeze(0).repeat(src.shape[0],1, src.shape[1], 1, 1)
        distance = torch.cdist(src_expand, dst_expand)
        src2dst_distance = torch.min(distance, dim=3)[0]  # (num_q, num_gt, num_pts_N)
        dst2src_distance = torch.min(distance, dim=2)[0]  # (num_q, num_gt, num_pts_M)
        loss_src = (src2dst_distance * src_weight).mean(-1)
        loss_dst = (dst2src_distance * dst_weight).mean(-1)
        loss = loss_src*self.loss_src_weight + loss_dst * self.loss_dst_weight
        return loss

@mmcv.jit(derivate=True, coderize=True)
def chamfer_distance(src,
                     dst,
                     src_weight=1.0,
                     dst_weight=1.0,
                    #  criterion_mode='l1',
                     reduction='mean',
                     avg_factor=None):
    """Calculate Chamfer Distance of two sets.

    Args:
        src (torch.Tensor): Source set with shape [B, N, C] to
            calculate Chamfer Distance.
        dst (torch.Tensor): Destination set with shape [B, M, C] to
            calculate Chamfer Distance.
        src_weight (torch.Tensor or float): Weight of source loss.
        dst_weight (torch.Tensor or float): Weight of destination loss.
        criterion_mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are 'none', 'sum' or 'mean'.

    Returns:
        tuple: Source and Destination loss with the corresponding indices.

            - loss_src (torch.Tensor): The min distance \
                from source to destination.
            - loss_dst (torch.Tensor): The min distance \
                from destination to source.
            - indices1 (torch.Tensor): Index the min distance point \
                for each point in source to destination.
            - indices2 (torch.Tensor): Index the min distance point \
                for each point in destination to source.
    """

    # if criterion_mode == 'smooth_l1':
    #     criterion = smooth_l1_loss
    # elif criterion_mode == 'l1':
    #     criterion = l1_loss
    # elif criterion_mode == 'l2':
    #     criterion = mse_loss
    # else:
    #     raise NotImplementedError

    # src_expand = src.unsqueeze(2).repeat(1, 1, dst.shape[1], 1)
    # dst_expand = dst.unsqueeze(1).repeat(1, src.shape[1], 1, 1)
    # import pdb;pdb.set_trace()
    distance = torch.cdist(src, dst)
    src2dst_distance, indices1 = torch.min(distance, dim=2)  # (B,N)
    dst2src_distance, indices2 = torch.min(distance, dim=1)  # (B,M)
    # import pdb;pdb.set_trace()
    #TODO this may be wrong for misaligned src_weight, now[N,fixed_num]
    # should be [N], then view
    loss_src = (src2dst_distance * src_weight)
    loss_dst = (dst2src_distance * dst_weight)
    if avg_factor is None:
        reduction_enum = F._Reduction.get_enum(reduction)
        if reduction_enum == 0:
            raise ValueError('MyCDLoss can not be used with reduction=`none`')
        elif reduction_enum == 1:
            loss_src = loss_src.mean(-1).mean()
            loss_dst = loss_dst.mean(-1).mean()
        elif reduction_enum == 2:
            loss_src = loss_src.mean(-1).sum()
            loss_dst = loss_dst.mean(-1).sum()
        else:
            raise NotImplementedError
    else:
        if reduction == 'mean':
            eps = torch.finfo(torch.float32).eps
            loss_src = loss_src.mean(-1).sum() / (avg_factor + eps)
            loss_dst = loss_dst.mean(-1).sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')

    return loss_src, loss_dst, indices1, indices2


@LOSSES.register_module()
class MyChamferDistance(nn.Module):
    """Calculate Chamfer Distance of two sets.

    Args:
        mode (str): Criterion mode to calculate distance.
            The valid modes are smooth_l1, l1 or l2.
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_src_weight (float): Weight of loss_source.
        loss_dst_weight (float): Weight of loss_target.
    """

    def __init__(self,
                #  mode='l1',
                 reduction='mean',
                 loss_src_weight=1.0,
                 loss_dst_weight=1.0):
        super(MyChamferDistance, self).__init__()

        # assert mode in ['smooth_l1', 'l1', 'l2']
        assert reduction in ['none', 'sum', 'mean']
        # self.mode = mode
        self.reduction = reduction
        self.loss_src_weight = loss_src_weight
        self.loss_dst_weight = loss_dst_weight

    def forward(self,
                source,
                target,
                src_weight=1.0,
                dst_weight=1.0,
                avg_factor=None,
                reduction_override=None,
                return_indices=False,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            source (torch.Tensor): Source set with shape [B, N, C] to
                calculate Chamfer Distance.
            target (torch.Tensor): Destination set with shape [B, M, C] to
                calculate Chamfer Distance.
            src_weight (torch.Tensor | float, optional):
                Weight of source loss. Defaults to 1.0.
            dst_weight (torch.Tensor | float, optional):
                Weight of destination loss. Defaults to 1.0.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.
            return_indices (bool, optional): Whether to return indices.
                Defaults to False.

        Returns:
            tuple[torch.Tensor]: If ``return_indices=True``, return losses of \
                source and target with their corresponding indices in the \
                order of ``(loss_source, loss_target, indices1, indices2)``. \
                If ``return_indices=False``, return \
                ``(loss_source, loss_target)``.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        loss_source, loss_target, indices1, indices2 = chamfer_distance(
            source, target, src_weight, dst_weight, reduction,
            avg_factor=avg_factor)

        loss_source *= self.loss_src_weight
        loss_target *= self.loss_dst_weight

        loss_pts = loss_source + loss_target

        if return_indices:
            return loss_pts, indices1, indices2
        else:
            return loss_pts




#######################################################################
# Losses / Matching costs used for rasterization-based supervison
#######################################################################


# for matching only!
def batch_dice_cost(inputs, targets, already_sigmoided=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if not already_sigmoided:
        inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


# For matching only! 
def batch_sigmoid_focal_cost(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    prob = inputs.sigmoid()
    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    focal_neg = (prob ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )
    if alpha >= 0:
        focal_pos = focal_pos * alpha
        focal_neg = focal_neg * (1 - alpha)

    loss = torch.einsum("nc,mc->nm", focal_pos, targets) + torch.einsum(
        "nc,mc->nm", focal_neg, (1 - targets)
    )

    return loss / hw


# for instance segmentation loss computation only!
def mask_dice_loss(inputs, targets, num_masks, already_sigmoided=False):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if not already_sigmoided:
        inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)

    return loss.sum() / num_masks

    # # NOTE: we don't reduce dimension here, because we still need to perform 
    # #       element-wise minimum on the loss.
    # return loss / num_masks


# for instance segmentation loss computation only!
def mask_sigmoid_focal_loss(inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks



@MATCH_COST.register_module()
class RenderedMaskDiceCost(object):
    def __init__(self, weight=10.):
        self.weight = weight

        # TODO: to make rasterization's hyperparameters configurable and not hardcoded
        self.renderer_lane = SoftLane(2.0, 'boundary')
        self.renderer_polygon = SoftPolygon(4.0, 'mask')
        self.renderer_H = 128
        self.renderer_W = 64
        self.polygon_class_labels = [1]
        self.lane_class_labels = [0, 2]

    @torch.no_grad()
    def __call__(self, pred_cls, pred_pts, gt_labels, gt_pts):
        num_preds = pred_cls.shape[0]
        num_gts = gt_labels.shape[0]
        num_pts_per_pred = pred_pts.shape[1]
        num_pts_per_gt = gt_pts.shape[1]

        assert pred_cls.shape[0] == pred_pts.shape[0]
        assert gt_labels.shape[0] == gt_pts.shape[0]
        assert pred_pts.shape[-1] == 2
        assert gt_pts.shape[-1] == 2

        # unnormalize the coordinates
        pred_pts = pred_pts * torch.tensor([self.renderer_W, self.renderer_H], device=pred_pts.device).reshape(1,1,2)
        gt_pts = gt_pts * torch.tensor([self.renderer_W, self.renderer_H], device=gt_pts.device).reshape(1,1,2)

        # rasterize the predicted points
        pred_rendered_as_lanes = self.renderer_lane(pred_pts, self.renderer_W, self.renderer_H)
        pred_rendered_as_polygons = self.renderer_polygon(pred_pts, self.renderer_W, self.renderer_H)

        # rasterize the ground truth points
        gt_rendered_as_lanes = self.renderer_lane(gt_pts, self.renderer_W, self.renderer_H)
        gt_rendered_as_polygons = self.renderer_polygon(gt_pts, self.renderer_W, self.renderer_H)

        # based on gt_labels, select rasterized ground truth points
        gt_selection_indexes = [1 if gt_labels[i] in self.polygon_class_labels else 0 for i in range(num_gts)]
        rendered_gt = torch.zeros((num_gts, self.renderer_H, self.renderer_W), device=gt_pts.device)
        for i in range(num_gts):
            rendered_gt[i] = gt_rendered_as_polygons[i] if (gt_selection_indexes[i] == 1) else gt_rendered_as_lanes[i]
        
        # calculate the cost matrix
        cost_line = batch_dice_cost(pred_rendered_as_lanes.reshape(-1, self.renderer_H*self.renderer_W),
                                    rendered_gt.reshape(-1, self.renderer_H*self.renderer_W), already_sigmoided=True)

        cost_polygon = batch_dice_cost(pred_rendered_as_polygons.reshape(-1, self.renderer_H*self.renderer_W),
                                       rendered_gt.reshape(-1, self.renderer_H*self.renderer_W), already_sigmoided=True)

        cost_matrix = torch.zeros_like(cost_line)
        for i in range(num_gts):
            cost_matrix[:, i] = cost_polygon[:, i] if (gt_labels[i] in self.polygon_class_labels) else cost_line[:, i]
        
        return cost_matrix * self.weight


@LOSSES.register_module()
class RenderedMaskDiceLoss(object):
    def __init__(self, weight=10.):
        self.weight = weight

        # TODO: to make rasterization's hyperparameters configurable and not hardcoded
        self.renderer_lane = SoftLane(2.0, 'boundary')
        self.renderer_polygon = SoftPolygon(4.0, 'mask')
        self.renderer_H = 256
        self.renderer_W = 128
        self.polygon_class_labels = [1]
        self.lane_class_labels = [0, 2]

    def __call__(self, pred_cls, pred_pts, gt_labels, gt_pts):
        num_preds = pred_cls.shape[0]
        num_gts = gt_labels.shape[0]
        num_pts_per_pred = pred_pts.shape[1]
        num_pts_per_gt = gt_pts.shape[1]

        assert pred_cls.shape[0] == pred_pts.shape[0]
        assert gt_labels.shape[0] == gt_pts.shape[0]
        assert pred_pts.shape[-1] == 2
        assert gt_pts.shape[-1] == 2

        # unnormalize the coordinates
        pred_pts = pred_pts * torch.tensor([self.renderer_W, self.renderer_H], device=pred_pts.device).reshape(1,1,2)
        gt_pts = gt_pts * torch.tensor([self.renderer_W, self.renderer_H], device=gt_pts.device).reshape(1,1,2)

        # rasterize the predicted points
        pred_rendered_as_lanes = self.renderer_lane(pred_pts, self.renderer_W, self.renderer_H)
        pred_rendered_as_polygons = self.renderer_polygon(pred_pts, self.renderer_W, self.renderer_H)

        with torch.no_grad():
            # rasterize the ground truth points
            gt_rendered_as_lanes = self.renderer_lane(gt_pts, self.renderer_W, self.renderer_H)
            gt_rendered_as_polygons = self.renderer_polygon(gt_pts, self.renderer_W, self.renderer_H)

            # based on gt_labels, select rasterized ground truth points
            gt_selection_indexes = [1 if gt_labels[i] in self.polygon_class_labels else 0 for i in range(num_gts)]
            rendered_gt = torch.zeros((num_gts, self.renderer_H, self.renderer_W), device=gt_pts.device)
            for i in range(num_gts):
                rendered_gt[i] = gt_rendered_as_polygons[i] if (gt_selection_indexes[i] == 1) else gt_rendered_as_lanes[i]

        # select rendering predictions to produce loss based on matching results
        pred_rendered_masks = torch.zeros_like(pred_rendered_as_lanes)
        for i in range(num_preds):
            pred_rendered_masks[i, ...] = pred_rendered_as_polygons[i, ...] if (gt_labels[i] in self.polygon_class_labels) else pred_rendered_as_lanes[i, ...]

        
        # # calculate the loss
        # loss1 = mask_dice_loss(pred_rendered_as_lanes.reshape(-1, self.renderer_H*self.renderer_W), 
        #                        rendered_gt.reshape(-1, self.renderer_H*self.renderer_W),
        #                        num_masks=num_gts,
        #                        already_sigmoided=True)
        
        # loss2 = mask_dice_loss(pred_rendered_as_polygons.reshape(-1, self.renderer_H*self.renderer_W),
        #                        rendered_gt.reshape(-1, self.renderer_H*self.renderer_W),
        #                        num_masks=num_gts,
        #                        already_sigmoided=True)
        
        # # compute the element-wise minimum of the two losses
        # loss = torch.minimum(loss1, loss2)
        # loss = loss.sum()

        loss = mask_dice_loss(pred_rendered_masks.reshape(-1, self.renderer_H*self.renderer_W), 
                              rendered_gt.reshape(-1, self.renderer_H*self.renderer_W),
                              num_masks=num_gts,
                              already_sigmoided=True)
        
        return loss * self.weight