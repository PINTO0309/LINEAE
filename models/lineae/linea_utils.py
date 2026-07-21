import torch 
import torch.nn as nn
import torch.nn.functional as F


def select_top_line_predictions(logits, lines, num_select):
    """Select evaluator-aligned class-0 predictions without changing training output."""
    exporting = torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()
    if not exporting:
        if logits.ndim != 3 or lines.ndim != 3 or lines.shape[-1] != 4:
            raise ValueError("line selection expects [B,Q,C] logits and [B,Q,4] lines")
        if logits.shape[:2] != lines.shape[:2]:
            raise ValueError("line selection logits/lines batch-query shapes must match")
    num_select = int(num_select)
    if num_select <= 0:
        raise ValueError("num_select must be positive")
    if not exporting:
        if num_select > logits.shape[1]:
            raise ValueError(
                f"num_select={num_select} exceeds query count {logits.shape[1]}"
            )
        if num_select == logits.shape[1]:
            return logits, lines
    indices = torch.topk(logits[..., 0], num_select, dim=1, sorted=True).indices
    return (
        torch.gather(logits, 1, indices.unsqueeze(-1).expand(-1, -1, logits.shape[-1])),
        torch.gather(lines, 1, indices.unsqueeze(-1).expand(-1, -1, 4)),
    )


def endpoint_swap(lines):
    if lines.shape[-1] != 4:
        raise ValueError(f"line tensors must end in four coordinates, got {tuple(lines.shape)}")
    return lines[..., [2, 3, 0, 1]]


def endpoint_invariant_loss(direct, swapped):
    """Select the cheaper endpoint order with a deterministic gradient on ties.

    ``torch.minimum`` splits the gradient between equal inputs.  For a
    zero-length prediction, directed and swapped line losses are exactly equal,
    and that split gives both predicted endpoints identical gradients.  LINEA's
    point anchors then cannot acquire a non-zero length.  Selecting the direct
    branch on ties keeps the same endpoint-invariant scalar value while breaking
    that optimization symmetry.
    """
    return torch.where(direct <= swapped, direct, swapped)


def pairwise_endpoint_l1(student_lines, target_lines):
    """Pairwise undirected-line L1 cost."""
    direct = torch.cdist(student_lines.float(), target_lines.float(), p=1)
    swapped = torch.cdist(student_lines.float(), endpoint_swap(target_lines).float(), p=1)
    return torch.minimum(direct, swapped)


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        reg_max (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(reg_max/2)=0
                           and steeper weights at both ends.
        deploy (bool): If True, uses deployment mode settings.

    Returns:
        Tensor: Sequence of Weighting Function.
    """
    if deploy:
        if up.device.type == "meta":
            # Deployment FLOP accounting for very large variants runs the
            # complete graph on meta tensors. Values do not affect graph
            # structure; only the reg_max + 1 lookup-table shape is required.
            return torch.empty(reg_max + 1, dtype=up.dtype, device=up.device)
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.cat(values, 0)


def translate_gt(gt, reg_max, reg_scale, up):
    """
    Decodes bounding box ground truth (GT) values into distribution-based GT representations.

    This function maps continuous GT values into discrete distribution bins, which can be used
    for regression tasks in object detection models. It calculates the indices of the closest
    bins to each GT value and assigns interpolation weights to these bins based on their proximity
    to the GT value.

    Args:
        gt (Tensor): Ground truth bounding box values, shape (N, ).
        reg_max (int): Maximum number of discrete bins for the distribution.
        reg_scale (float): Controls the curvature of the Weighting Function.
        up (Tensor): Controls the upper bounds of the Weighting Function.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - indices (Tensor): Index of the left bin closest to each GT value, shape (N, ).
            - weight_right (Tensor): Weight assigned to the right bin, shape (N, ).
            - weight_left (Tensor): Weight assigned to the left bin, shape (N, ).
    """
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)

    # Find the closest left-side indices for each value
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1

    # Calculate the weights for the interpolation
    indices = closest_left_indices.float()

    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)

    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()

    # Obtain distances
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]

    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])

    # Valid weights
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]

    # Invalid weights (out of range)
    invalid_idx_mask_neg = (indices < 0)
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0

    invalid_idx_mask_pos = (indices >= reg_max)
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1

    return indices, weight_right, weight_left


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """
    Converts bounding box coordinates to distances from a reference point.

    Args:
        points (Tensor): (n, 4) [x, y, w, h], where (x, y) is the center.
        bbox (Tensor): (n, 4) bounding boxes in "xyxy" format.
        reg_max (float): Maximum bin value.
        reg_scale (float): Controling curvarture of W(n).
        up (Tensor): Controling upper bounds of W(n).
        eps (float): Small value to ensure target < reg_max.

    Returns:
        Tensor: Decoded distances.
    """
    reg_scale = abs(reg_scale)

    Dx = torch.abs(points[..., 0] - points[..., 2])
    Dy = torch.abs(points[..., 1] - points[..., 3])

    left   = (points[:, 0] - bbox[:, 0]) / (Dx / reg_scale + 1e-16) - 0.5 * reg_scale
    top    = (points[:, 1] - bbox[:, 1]) / (Dy / reg_scale + 1e-16) - 0.5 * reg_scale
    right  = (points[:, 2] - bbox[:, 2]) / (Dx / reg_scale + 1e-16) - 0.5 * reg_scale
    bottom = (points[:, 3] - bbox[:, 3]) / (Dy / reg_scale + 1e-16) - 0.5 * reg_scale
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max-eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def distance2bbox(points, distance, reg_scale):
    """
    Decodes edge-distances into bounding box coordinates.

    Args:
        points (Tensor): (B, N, 4) or (N, 4) format, representing [x, y, w, h],
                         where (x, y) is the center and (w, h) are width and height.
        distance (Tensor): (B, N, 4) or (N, 4), representing distances from the
                           point to the left, top, right, and bottom boundaries.

        reg_scale (float): Controls the curvature of the Weighting Function.

    Returns:
        Tensor: Bounding boxes in (N, 4) or (B, N, 4) format [cx, cy, w, h].
    """
    reg_scale = abs(reg_scale)

    Dx = torch.abs(points[..., 0] - points[..., 2])
    Dy = torch.abs(points[..., 1] - points[..., 3])

    x1 = points[..., 0] + (0.5 * reg_scale + distance[..., 0]) * (Dx / reg_scale)
    y1 = points[..., 1] + (0.5 * reg_scale + distance[..., 1]) * (Dy / reg_scale)
    x2 = points[..., 2] + (0.5 * reg_scale + distance[..., 2]) * (Dx / reg_scale)
    y2 = points[..., 3] + (0.5 * reg_scale + distance[..., 3]) * (Dy / reg_scale)

    bboxes = torch.stack([x1, y1, x2, y2], -1)

    return bboxes

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
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
    prob =  inputs.sigmoid()  
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1-alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def get_activation(act: str, inplace: bool=True):
    """get activation
    """
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act 

    act = act.lower()
    
    if act == 'silu' or act == 'swish':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()

    elif act == 'hardsigmoid':
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inplace
    
    return m 
