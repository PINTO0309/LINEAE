# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from .linea_utils import inverse_sigmoid

def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    if training:
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        device = targets[0]['labels'].device
        # positive and negative dn queries
        dn_number = dn_number * 2
        known = [torch.ones_like(t['labels']) for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        lines = torch.cat([t['lines'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_lines = lines.repeat(2 * dn_number, 1)

        known_labels_expaned = known_labels.clone()
        known_lines_expand = known_lines.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        positive_idx = torch.arange(len(lines), device=device).long().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.arange(dn_number, device=device) * len(lines) * 2).long().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(lines)

        # Perturb each endpoint along the segment direction.  The inherited
        # LINEA expression rebuilt both endpoints around ``first_endpoint / 2``;
        # even with a zero noise scale it therefore replaced every target line
        # with a zero-length point.  Denoising noise must be additive so that a
        # zero scale is the identity and positive queries stay centred on their
        # corresponding target segment.
        diff = torch.zeros_like(known_lines)
        diff[:, :2] = (known_lines[:, 2:] -  known_lines[:, :2]) / 2
        diff[:, 2:] = (known_lines[:, 2:] -  known_lines[:, :2]) / 2

        rand_sign = torch.randint(low=0, high=2, size=(known_lines.shape[0], 2), dtype=torch.float32, device=known_lines.device) * 2.0 - 1.0
        rand_part = torch.rand(size=(known_lines.shape[0], 2), device=known_lines.device)
        rand_part[negative_idx] += 1.2 
        rand_part *= rand_sign

        known_lines_ = known_lines + torch.mul(
            rand_part.repeat_interleave(2, 1), diff
        ) * box_noise_scale

        known_lines_expand = known_lines_.clamp(min=0.0, max=1.0)

        # order: top point > bottom point
        #        if same y coordinate, right point > left point
    
        idx = torch.logical_or(known_lines_expand[..., 0] > known_lines_expand[..., 2],
                torch.logical_or(
                known_lines_expand[..., 0] == known_lines_expand[..., 2],
                known_lines_expand[..., 1] < known_lines_expand[..., 3]
                )
            )

        known_lines_expand[idx] = known_lines_expand[idx][:, [2, 3, 0, 1]]

        m = known_labels_expaned.long().to(device)
        input_label_embed = label_enc(m)
        input_lines_embed = inverse_sigmoid(known_lines_expand)

        padding_label = torch.zeros(pad_size, hidden_dim, device=device)
        padding_lines = torch.zeros(pad_size, 4, device=device)

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_lines = padding_lines.repeat(batch_size, 1, 1)

        map_known_indice = torch.empty(0, device=device)
        if len(known_num):
            map_known_indice = torch.cat([
                torch.arange(int(num), device=device) for num in known_num
            ])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()

        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_lines[(known_bid.long(), map_known_indice)] = input_lines_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool, device=device)
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_lines = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_lines, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        # print(output_known_class.shape, outputs_class.shape)
        # quit()
        out = {'pred_logits': output_known_class[-1], 'pred_lines': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class[1:], output_known_coord[1:])
            out['pre_outputs'] = {'pred_logits':output_known_class[0], 'pred_lines': output_known_coord[0]}
        dn_meta['output_known_lbs_lines'] = out
    return outputs_class, outputs_coord

