# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
from typing import Optional
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN) """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def gen_sineembed_for_2d_position(pos_tensor):
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_tx = 20 ** (2 * (dim_t // 2) / 128)
    dim_ty = 20 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_tx
    pos_y = y_embed[:, :, None] / dim_ty
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2)
    return pos


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False):

        super().__init__()

        self.num_sample = int(num_queries * 0.2)
        self.pts_per_sample = 8

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim,
                                          keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

        self.sampling_pt = MLP(self.d_model, self.d_model, self.pts_per_sample * (2 + 4), 2)
        nn.init.constant_(self.sampling_pt.layers[-1].weight.data, 0)
        nn.init.constant_(self.sampling_pt.layers[-1].bias.data, 0)

        self.num_dynamic = 2
        self.dim_dynamic = 64
        self.num_params = self.d_model * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.d_model, self.num_dynamic * self.num_params)
        self.dynamic_norm1 = nn.LayerNorm(self.dim_dynamic)
        self.dynamic_norm2 = nn.LayerNorm(self.d_model)

        self.level_embed = nn.Parameter(torch.Tensor(4, self.d_model))
        nn.init.normal_(self.level_embed)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, src, mask, refpoints_unsigmoid, pos_embed, coord, feature_pyramid, feature_pyramid_mask):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        memory_h, memory_w = h, w
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoints_unsigmoid = refpoints_unsigmoid.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        num_queries = refpoints_unsigmoid.shape[0]
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoints_unsigmoid.device)
        else:
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)  # (n_q*n_pat, bs, d_model)
            refpoints_unsigmoid = refpoints_unsigmoid.repeat(self.num_patterns, 1, 1)  # (n_q*n_pat, bs, d_model)

        assert len(self.encoder.layers) == len(self.decoder.layers)
        enc_output = src
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id in range(len(self.decoder.layers)):

            if layer_id != 0:
                sampling_locations = self.sampling_pt(sampled_output).reshape(self.num_sample, bs, self.pts_per_sample, (2 + 4))
                sampling_scales = F.softmax(sampling_locations[:, :, :, 2:], dim=-1)
                sampling_locations = sampling_locations[:, :, :, :2]
                sampling_locations = sampled_reference_points[:, :, :2].unsqueeze(2) + sampled_reference_points[:, :, 2:].unsqueeze(2) * sampling_locations.tanh()

                ms_supp_mem = []
                for feat_lvl in range(4):
                    memory_2d, memory_2d_mask = feature_pyramid[feat_lvl], feature_pyramid_mask[feat_lvl]
                    valid_ratio = self.get_valid_ratio(memory_2d_mask)
                    sampling_grids = (sampling_locations * valid_ratio.reshape(1, bs, 1, 2) * 2.0 - 1.0).permute(1, 0, 2, 3).reshape(bs, self.num_sample, self.pts_per_sample, 2)
                    supplementary_memory = F.grid_sample(memory_2d, sampling_grids, padding_mode='zeros', align_corners=False)
                    supplementary_memory = supplementary_memory.reshape(bs, self.d_model, self.num_sample * self.pts_per_sample).permute(2, 0, 1)
                    supplementary_memory = supplementary_memory + self.level_embed[feat_lvl].reshape(1, 1, self.d_model)
                    ms_supp_mem.append(supplementary_memory)
                ms_supp_mem = torch.stack(ms_supp_mem).permute(1, 2, 3, 0)
                sampling_scales = sampling_scales.permute(0, 2, 1, 3).reshape(self.num_sample * self.pts_per_sample, bs, 1, 4)

                supplementary_memory = (sampling_scales * ms_supp_mem).sum(dim=3, keepdim=False)
                supplementary_memory_coord = sampling_locations.permute(0, 2, 1, 3).reshape(self.num_sample * self.pts_per_sample, bs, 2)
                supplementary_memory_pos = gen_sineembed_for_2d_position(supplementary_memory_coord)
                supplementary_memory_mask = torch.zeros(bs, self.num_sample * self.pts_per_sample, dtype=torch.bool, device=src.device)

                # Dynamic layer for supp_mem
                parameters = self.dynamic_layer(sampled_output)
                param1 = parameters[:, :, :self.num_params].view(-1, self.d_model, self.dim_dynamic)
                param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.d_model)
                supplementary_memory = supplementary_memory.reshape(self.num_sample, self.pts_per_sample, bs, self.d_model)
                supplementary_memory = supplementary_memory.permute(0, 2, 1, 3).reshape(self.num_sample * bs, self.pts_per_sample, self.d_model)
                supplementary_memory = F.relu(self.dynamic_norm1(torch.bmm(supplementary_memory, param1)))
                supplementary_memory = self.dynamic_norm2(torch.bmm(supplementary_memory, param2))
                supplementary_memory = supplementary_memory.reshape(self.num_sample, bs, self.pts_per_sample, self.d_model)
                supplementary_memory = supplementary_memory.permute(0, 2, 1, 3).reshape(self.num_sample * self.pts_per_sample, bs, self.d_model)

                if layer_id == 1:
                    enc_output_ = torch.cat([enc_output, supplementary_memory], dim=0)
                else:
                    enc_output_ = torch.cat([enc_output + enc_output_first, supplementary_memory], dim=0)

                pos_embed_ = torch.cat([pos_embed, supplementary_memory_pos], dim=0)
                mask_ = torch.cat([mask, supplementary_memory_mask], dim=1)
                
                # ----------- Encoder -----------
                pos_scales = self.encoder.query_scale(enc_output_)
                enc_output_ = self.encoder.layers[layer_id](enc_output_, src_mask=None, src_key_padding_mask=mask_, pos=pos_embed_*pos_scales)
                if self.encoder.norm is not None:
                    enc_output_ = self.encoder.norm(enc_output_)

                decoder_memory = enc_output_
                decoder_memory_pos = pos_embed_
                decoder_mask = mask_

                enc_output = enc_output_[:-(self.num_sample * self.pts_per_sample), :, :]
                pos_embed = pos_embed_[:-(self.num_sample * self.pts_per_sample), :, :]
                mask = mask_[:, :-(self.num_sample * self.pts_per_sample)]

            else:
                # ----------- Encoder -----------
                pos_scales = self.encoder.query_scale(enc_output)
                enc_output = self.encoder.layers[layer_id](enc_output, src_mask=None, src_key_padding_mask=mask, pos=pos_embed*pos_scales)
                if self.encoder.norm is not None:
                    enc_output = self.encoder.norm(enc_output)

                enc_output_first = enc_output

                decoder_memory = enc_output
                decoder_memory_pos = pos_embed
                decoder_mask = mask

            #  ----------- Decoder -----------
            obj_center = reference_points[..., :self.decoder.query_dim]   # [num_queries, batch_size, 2]
            # get sine embedding for the query vector
            query_sine_embed = gen_sineembed_for_position(obj_center)
            query_pos = self.decoder.ref_point_head(query_sine_embed)

            # For the first decoder layer, we do not apply transformation over p_s
            if self.decoder.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1.0
                else:
                    pos_transformation = self.decoder.query_scale(output)
            else:
                pos_transformation = self.decoder.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            # modulated HW attentions
            if self.decoder.modulate_hw_attn:
                refHW_cond = self.decoder.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            output = self.decoder.layers[layer_id](
                output,
                decoder_memory,
                memory_key_padding_mask=decoder_mask,
                pos=decoder_memory_pos,
                query_pos=query_pos,
                query_sine_embed=query_sine_embed,
                is_first=(layer_id == 0)
            )

            # iter update
            if self.decoder.bbox_embed is not None:
                if self.decoder.bbox_embed_diff_each_layer:
                    tmp = self.decoder.bbox_embed[layer_id](output)
                else:
                    tmp = self.decoder.bbox_embed(output)

                tmp[..., :self.decoder.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.decoder.query_dim].sigmoid()
                if layer_id != self.decoder.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

                scores = self.class_embed(output).max(dim=2, keepdim=False)[0]
                _, top_score_indices = torch.topk(scores, self.num_sample, dim=0)
                sampled_output = torch.zeros(self.num_sample, bs, self.d_model, device=output.device)
                for b in range(bs):
                    sampled_output[:, b, :] = output[top_score_indices[:, b], b, :]

                sampled_reference_points = torch.zeros(self.num_sample, bs, 4, device=reference_points.device)
                for b in range(bs):
                    sampled_reference_points[:, b, :] = reference_points[top_score_indices[:, b], b, :]

            if self.decoder.return_intermediate:
                intermediate.append(self.decoder.norm(output))

        if self.decoder.return_intermediate:
            if self.decoder.bbox_embed is not None:
                hs = torch.stack(intermediate).transpose(1, 2)
                references = torch.stack(ref_points).transpose(1, 2)
            else:
                hs = torch.stack(intermediate).transpose(1, 2)
                references = reference_points.unsqueeze(0).transpose(1, 2)

        return hs, references


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)

        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.d_model = d_model
        self.nhead = nhead

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                      src_mask: Optional[Tensor] = None,
                      src_key_padding_mask: Optional[Tensor] = None,
                      pos: Optional[Tensor] = None,):

        # q = k = self.with_pos_embed(src, pos)
        # src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]

        num_queries, bs = src.shape[0], src.shape[1]

        q_content = self.sa_qcontent_proj(src)
        k_content = self.sa_kcontent_proj(src)
        v = self.sa_v_proj(src)
        q_pos = self.sa_qpos_proj(pos)
        k_pos = self.sa_kpos_proj(pos)

        q_content = q_content.view(num_queries, bs, self.nhead, self.d_model // self.nhead)
        q_pos = q_pos.view(num_queries, bs, self.nhead, self.d_model // self.nhead)
        # q = torch.cat([q_content, q_pos], dim=3).view(num_queries, bs, self.d_model * 2)
        q = (q_content + q_pos).view(num_queries, bs, self.d_model)
        k_content = k_content.view(num_queries, bs, self.nhead, self.d_model // self.nhead)
        k_pos = k_pos.view(num_queries, bs, self.nhead, self.d_model // self.nhead)
        # k = torch.cat([k_content, k_pos], dim=3).view(num_queries, bs, self.d_model * 2)
        k = (k_content + k_pos).view(num_queries, bs, self.d_model)
        src2 = self.self_attn(query=q,
                              key=k,
                              value=v,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + tgt2
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
