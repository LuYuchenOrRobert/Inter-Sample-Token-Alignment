# -*- coding: utf-8 -*-
"""
Copyright (c) 2025 the Authors of the Paper 'Inter-Sample Token Alignment for
Vision Transformers' submitted to ICCV 2025. All rights reserved.

References:
    * https://github.com/facebookresearch/moco-v3
"""

import math
import torch
import torch.nn as nn
import warnings


def concat_all_gather(tensor, dim=0):
    # follows MoCo-v3: https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensor = tensor.contiguous()
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # This extra code is designed for the `k` in `ISTA2` module which does require gradients - by Authors of ISTA2
    # https://amsword.medium.com/gradient-backpropagation-with-torch-distributed-all-gather-9f3941a381f8
    tensors_gather[torch.distributed.get_rank()] = tensor

    output = torch.cat(tensors_gather, dim=dim)
    return output


class ISTA2(nn.Module):
    def __init__(self, dim, num_heads=8, qk_scale=1., v_scale=1., attn_drop_layer=None, is_swin=False,
                 qk_norm=False, v_norm=False, norm_type='rmsnorm', ista2_method=None, attn_inter_topk=0,
                 rep_tkn_type=None, rep_tkn_ratio=None, cls_tkn_idx=None, bg_tkn_idx=None,
                 bs_ratio=None, across_gpus=False):
        """
        Inter-sample Token Alignment Attention (ISTA2) mechanism

        Args
        ----------
            dim : int
                the dimension of patch/token embeddings
            num_heads : int
                the number of attention heads
            qk_scale : float
                scale factor for the dot product between query and key tensors
            v_scale : float
                scale factor for the dot product between value tensors
            attn_drop_layer : torch.nn.Module
                dropout layer for attention matrix
            is_swin : bool
                whether the model is a Swin Transformer
            qk_norm : bool
                whether to apply normalisation to the query and key tensors
            v_norm : bool
                whether to apply normalisation to the value tensor
            norm_type : str
                'layernorm': Layer Normalisation is applied
                'rmsnorm': Root Mean Square Layer Normalisation is applied
            ista2_method : str or None
                None: standard self-attention mechanism is applied
                'v': inter-sample token alignment attention mechanism is applied
            attn_inter_topk : int
                the number of top-k tokens to be selected in the inter-sample token alignment attention matrix
            rep_tkn_type : str or None
                the type of representative token for each sample serving as the reference sample
                None: all original tokens are used for each sample
                'cls': the class token is used as the representative token for each sample
                'cls+bg': the class and background tokens are used as the representative tokens for each sample
                'mean': the global average pooling is used to get the representative token for each sample
                'random': randomly select some tokens as the representative tokens for each sample according to
                    the 'rep_tkn_ratio' parameter
            rep_tkn_ratio : float or None
                the ratio of the number of representative tokens to the total number of tokens in each sample when
                'rep_tkn_type' is 'random'
            cls_tkn_idx : int
                index of the class token
            bg_tkn_idx : int
                index of the background token
            bs_ratio : float or None
                the ratio of the number of reference samples to mini-batch size (or batch size if `across_gpus` is True)
            across_gpus : bool
                whether to gather the reference samples across all gpus (only for ISTA2 mechanism)
        Returns
        ----------
            ISTA2 : torch.nn.Module

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        super(ISTA2, self).__init__()

        assert norm_type in ['layernorm', 'rmsnorm'], f"norm_type(={norm_type}) must be 'layernorm' or 'rmsnorm'."
        assert ista2_method in [None, 'v'], f"ista2_method(={ista2_method}) must be None or 'v'."
        assert isinstance(attn_inter_topk, int) and attn_inter_topk >= 0, (
            f"attn_inter_topk(={attn_inter_topk}) must be a non-negative integer."
        )
        if attn_inter_topk == 0:
            assert ista2_method is None, (
                f"attn_inter_topk(={attn_inter_topk}) must be greater than 0 under the settings of "
                f"ista2_method(={ista2_method})."
            )
        assert rep_tkn_type in [None, 'cls', 'cls+bg', 'mean', 'random'], (
            f"rep_tkn_type(={rep_tkn_type}) must be None, 'cls', 'cls+bg', 'mean' or 'random'."
        )
        if rep_tkn_type == 'cls':
            assert isinstance(cls_tkn_idx, int), (
                f"cls_tkn_idx(={cls_tkn_idx}) must be an integer when rep_tkn_type='cls'."
            )
        elif rep_tkn_type == 'cls+bg':
            assert isinstance(cls_tkn_idx, int) and isinstance(bg_tkn_idx, int), (
                f"cls_tkn_idx(={cls_tkn_idx}) and bg_tkn_idx(={bg_tkn_idx}) "
                f"must be integers when rep_tkn_type='cls+bg'."
            )
        elif rep_tkn_type == 'random':
            assert isinstance(rep_tkn_ratio, float) and 0 < rep_tkn_ratio <= 1, (
                f"rep_tkn_ratio(={rep_tkn_ratio}) must be a float in the range of (0, 1] when rep_tkn_type='random'."
            )
        else:
            assert rep_tkn_ratio is None and cls_tkn_idx is None and bg_tkn_idx is None, (
                f"rep_tkn_ratio(={rep_tkn_ratio}), cls_tkn_idx(={cls_tkn_idx}) and bg_tkn_idx(={bg_tkn_idx}) "
                f"must all be None when rep_tkn_type is set to 'mean' or None."
            )
        if bs_ratio is not None:
            assert isinstance(bs_ratio, float) and 0 < bs_ratio <= 1, (
                f"bs_ratio(={bs_ratio}) must be a float in the range of (0, 1]."
            )

        self.num_heads = num_heads
        self.qk_scale = qk_scale
        self.v_scale = v_scale
        self.attn_drop_layer = attn_drop_layer
        self.is_swin = is_swin
        self.qk_norm = qk_norm
        self.v_norm = v_norm
        self.ista2_method = ista2_method
        self.attn_inter_topk = attn_inter_topk
        self.rep_tkn_type = rep_tkn_type
        self.rep_tkn_ratio = rep_tkn_ratio
        self.cls_tkn_idx = cls_tkn_idx
        self.bg_tkn_idx = bg_tkn_idx
        self.bs_ratio = bs_ratio
        self.across_gpus = across_gpus
        self.random_tkn_idxs = None

        # check whether running in a distributed manner
        if torch.distributed.is_available():
            self.distributed = True if torch.distributed.is_initialized() else False
        else:
            self.distributed = False

        if self.across_gpus and not self.distributed:
            self.across_gpus = False
            warnings.warn("The 'across_gpus' parameter is forcibly set to False in a non-distributed environment.")

        if self.qk_norm or self.v_norm:
            norm_layer = nn.RMSNorm if norm_type == 'rmsnorm' else nn.LayerNorm
            if self.qk_norm:
                self.q_norm_layer = norm_layer(dim)
                self.k_norm_layer = norm_layer(dim)
            if self.v_norm:
                self.v_norm_layer = norm_layer(dim)

    @staticmethod
    def get_representative_tokens(x, rep_tkn_type, cls_tkn_idx, bg_tkn_idx, random_tkn_idxs):
        """
        Args
        ----------
            x : torch.Tensor
                input tensor with shape of (..., p, d)
            rep_tkn_type : str or None
                the type of representative token for each sample serving as the reference sample
                None: all original tokens are used for each sample
                'cls': the class token is used as the representative token for each sample
                'cls+bg': the class and background tokens are used as the representative tokens for each sample
                'mean': the global average pooling is used to get the representative token for each sample
                'random': randomly select some tokens as the representative tokens for each sample according to
                    the 'rep_tkn_ratio' parameter
            cls_tkn_idx : int or None
                index of the class token
            bg_tkn_idx : int or None
                index of the background token
            random_tkn_idxs : torch.Tensor or None
                the indices of the randomly selected tokens

        Returns
        ----------
            x : torch.Tensor
                representative tokens with shape of (..., p', d), where p' <= p

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        if rep_tkn_type == 'cls':
            # use the cls token as the representative token for each sample
            x = x[..., [cls_tkn_idx], :]
        elif rep_tkn_type == 'cls+bg':
            # use the cls and bg tokens as the representative tokens for each sample
            x = x[..., [cls_tkn_idx, bg_tkn_idx], :]
        elif rep_tkn_type == 'mean':
            # use global average pooling to get the representative token for each sample
            x = x.mean(dim=-2, keepdim=True)
        elif rep_tkn_type == 'random':
            # use the randomly selected tokens as the representative tokens for each sample
            x = x[..., random_tkn_idxs, :]
        else:
            # rep_tkn_type: None, i.e., use all original tokens instead of representative tokens
            pass

        return x

    @staticmethod
    def get_sample_indices(m, across_gpus):
        """
        Args
        ----------
            m : int
                the number of samples in the mini-batch (i.e., the samples on the current gpu)
            across_gpus : bool
                whether the source samples are gathered across all gpus

        Returns
        ----------
            idxs : torch.Tensor
                the indices of each sample (on the current gpu) in the batch
        """
        if across_gpus:
            idxs = (torch.arange(m, dtype=torch.long) + m * torch.distributed.get_rank()).cuda()
        else:
            idxs = torch.arange(m, dtype=torch.long).cuda()

        return idxs

    @staticmethod
    def attn_topk_selection(attn, topk):
        """
        Args
        ----------
            attn : torch.Tensor
                input attention matrix with shape of (..., p1, p2)
            topk : int
                the number of top-k tokens to be selected along the last axis in the attention matrix

        Returns
        ----------
            attn : torch.Tensor
                the attention matrix with shape of (..., p1, p2) after top-k selection

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        index = torch.topk(attn, k=topk, dim=-1).indices
        topk_mask = torch.zeros_like(attn, device=attn.device).scatter_(-1, index, 1.)
        attn = torch.where(topk_mask > 0, attn, torch.full_like(attn, float('-inf')))

        return attn

    def _update_random_token_indices(self, p, device):
        """
        Args
        ----------
            p : int
                the number of tokens in each sample
            device : torch.device
                the device where the random token indices are stored

        Returns
        ----------
            None

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        p_rep = math.ceil(p * self.rep_tkn_ratio)
        self.random_tkn_idxs = torch.randperm(p, device=device)[:p_rep].sort().values

    def get_reference_tokens(self, x_perm, m, across_gpus, auxiliary_data=False):
        """
        In the inter-sample token alignment attention mechanism, alongside standard self-attention computations, an 
        auxiliary attention process operates across samples to align tokens from one sample with those of another. The 
        `get_reference_tokens` function accepts the `x_perm` tensor as input and produces a tensor that typically serves
        as the key or value tensor for this cross-sample attention. The `x_perm` tensor comprises either the samples 
        identical to those in the query tensor (with matching order) or the samples from auxiliary data (external to the
        batch). The resulting tensor must assemble a reference token set for each query sample, sourced either from 
        intra-batch samples or auxiliary data. In the first scenario, where reference tokens originate from intra-batch 
        samples, the identical ordering of `x_perm` and the query tensor necessitates excluding the query sample's own 
        tokens from the reference set to prevent redundancy. In the second scenario, auxiliary data samples generally 
        exhibit no overlap with the query tensor, eliminating the need for exclusion.

        Args
        ----------
            x_perm : torch.Tensor
                input tensor with permuted shape of (h, b, p, d) serving as the key or value tensor
            m : int
                the number of samples in the tensor serving as the query tensor
            across_gpus : bool
                whether to gather the reference samples across all gpus
            auxiliary_data : bool
                whether the samples in x_perm are from the auxiliary data (external to the batch).

        Returns
        ----------
            x_perm : torch.Tensor
                output tensor with shape of (h, m, n*j, d) (if auxiliary_data is True) or (h, m, (n - 1)*j, d) (if not)
                containing the reference tokens for each sample in the query tensor

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        # get the representative tokens
        # x_perm: (h, b, p, d) -> (h, b, j, d)
        x_perm = self.get_representative_tokens(x_perm, self.rep_tkn_type, self.cls_tkn_idx, self.bg_tkn_idx,
                                                self.random_tkn_idxs)

        # gather all source inputs
        # x_perm: (h, b, j, d) -> (h, n, j, d)
        if across_gpus and not auxiliary_data:
            x_perm = concat_all_gather(x_perm, dim=1)

        h, n, j, d = x_perm.shape

        # x_perm: (h, n, j, d) -> (m, h, n, j, d) -> (h, m, n, j, d)
        x_perm = x_perm.repeat(m, 1, 1, 1, 1).permute(1, 0, 2, 3, 4)

        if not auxiliary_data:
            # the indices of the samples (on the current gpu) in the batch
            idxs = self.get_sample_indices(m, across_gpus)
            # use mask to get the off-diagonal elements (i.e., between one sample and another) of x_perm
            # along axes m and n (PLEASE NOTE THAT m corresponding to the data sink must be in the FIRST dimension,
            # otherwise the mask and reshape operations will make a wrong group of n-1 samples for sample m!!!)
            mask_inter = torch.ones(m, n, dtype=torch.bool).cuda()
            mask_inter[:, idxs] = ~torch.eye(m, dtype=torch.bool).cuda()
            n = n - 1
            x_perm = x_perm[:, mask_inter, ...].reshape(h, m, n, j, d)

        if self.bs_ratio is not None:
            # get the number of source samples for each sample
            n = math.ceil(n * self.bs_ratio)
            # select the first n samples as the source samples for each sample
            # no need to shuffle the source samples because the data sampler would have already done this if needed
            x_perm = x_perm[:, :, :n, ...]

        x_perm = x_perm.reshape(h, m, n * j, d)

        return x_perm

    def self_attn(self, q_perm, k_perm, relative_position_bias=None, mask=None):
        """
        Args
        ----------
            q_perm : torch.Tensor
                query tensor with permuted shape of (h, b, p1, d)
            k_perm : torch.Tensor
                key tensor with permuted shape of (h, b, p2, d)
            relative_position_bias : torch.Tensor or None
                (Swin Transformer) learnable relative position bias tensor with shape of (h, p1, p2)
            mask : torch.Tensor or None
                (Swin Transformer) (0/-inf) mask tensor with shape of (num_windows, p1, p2)

        Returns
        ----------
            attn_self : torch.Tensor
                the self attention matrix with shape of (h, b, p1, p2)

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        # compute the self attention matrix
        # attn_self: (h, b, p1, p2)
        attn_self = (q_perm @ k_perm.transpose(-2, -1)) * self.qk_scale
        if self.rep_tkn_type == 'cls+bg':
            # calculate similarity measures for background token based on those of class token
            attn_self[..., self.bg_tkn_idx, :] = -1. * attn_self[..., self.cls_tkn_idx, :].detach().clone()
        h, b, p1, p2 = attn_self.shape

        # apply relative position bias and mask for Swin Transformer
        if relative_position_bias is not None:
            attn_self = attn_self + relative_position_bias.unsqueeze(1)
        if mask is not None:
            nw = mask.shape[0]
            attn_self = attn_self.reshape(h, b // nw, nw, p1, p2) + mask.unsqueeze(0).unsqueeze(0)
            attn_self = attn_self.reshape(h, b, p1, p2)

        return attn_self

    def ista2_v(self, v_as_q_perm, v_as_kv_perm):
        """
        Args
        ----------
            v_as_q_perm : torch.Tensor
                input value tensor with permuted shape of (h, b, p2, d) serving as the query tensor
            v_as_kv_perm : torch.Tensor
                input value tensor with permuted shape of (h, b, p2, d) serving as the key and value tensor

        Returns
        ----------
            v_perm : torch.Tensor
                the aligned value tensor with permuted shape of (h, b, p2, d)

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        b = v_as_q_perm.shape[1]

        # get the source tokens in v_as_kv_perm for each token of each sample in v_as_q_perm
        # v_as_kv_perm: (h, b, p2, d) -> (h, b, n*j, d)
        v_as_kv_perm = self.get_reference_tokens(v_as_kv_perm, b, self.across_gpus, auxiliary_data=False)

        # compute the scaled cosine similarity between each token and itself
        # attn_tkn: (h, b, p2, 1)
        attn_tkn = torch.square(v_as_q_perm).sum(dim=-1, keepdim=True) * self.v_scale
        # compute the scaled cosine similarity between each token and the tokens from other samples
        # attn_inter: (h, b, p2, n*j)
        attn_inter = (v_as_q_perm @ v_as_kv_perm.transpose(-2, -1)) * self.v_scale

        # apply top-k selection to attn_inter
        attn_inter = self.attn_topk_selection(attn_inter, self.attn_inter_topk)

        # concatenate attn_tkn and attn_inter along the last axis
        # attn: (h, b, p2, 1 + n*j)
        attn = torch.cat((attn_tkn, attn_inter), dim=-1)

        # apply softmax to attn along the last axis
        attn = attn.softmax(dim=-1)
        # compute the aligned value tensor
        # v_perm: (h, b, p2, d)
        v_perm = attn[..., :1] * v_as_q_perm + attn[..., 1:] @ v_as_kv_perm

        return v_perm

    def forward(self, q, k, v, real_b=None, relative_position_bias=None, mask=None):
        """
        Args
        ----------
            q : torch.Tensor
                query tensor with shape of (b, p1, h*d)
            k : torch.Tensor
                key tensor with shape of (b, p2, h*d)
            v : torch.Tensor
                value tensor with shape of (b, p2, h*d)
            real_b : int or None
                (Swin Transformer) the real batch size (i.e., the number of samples in the current gpu)
            relative_position_bias : torch.Tensor or None
                (Swin Transformer) learnable relative position bias tensor with shape of (h, p1, p2)
            mask : torch.Tensor or None
                (Swin Transformer) (0/-inf) mask tensor with shape of (num_windows, p1, p2)

        Returns
        ----------
            x : torch.Tensor
                extracted embedding tensor with shape of (b, p, h*d)

        Notations
        ----------
            b/b1/b2/m/n: batches, p/p'/p1/p2/i/j: patches/tokens, h: heads, d: head_dim
        """
        # apply the normalisation layer to q, k and v
        if self.qk_norm:
            q = self.q_norm_layer(q)
            k = self.k_norm_layer(k)
        if self.v_norm:
            v = self.v_norm_layer(v)

        b, p1, dim = q.shape
        _, p2, _ = k.shape
        d = dim // self.num_heads
        multi_samples = (real_b > 1) if self.is_swin else (b > 1)

        # q: (b, p1, h*d) -> (h, b, p1, d)
        q = q.reshape(b, p1, self.num_heads, d).permute(2, 0, 1, 3)
        # k & v: (b, p2, h*d) -> (h, b, p2, d)
        k = k.reshape(b, p2, self.num_heads, d).permute(2, 0, 1, 3)
        v = v.reshape(b, p2, self.num_heads, d).permute(2, 0, 1, 3)

        # compute the self attention matrix
        # attn: (h, b, p1, p2)
        attn = self.self_attn(q, k, relative_position_bias, mask)

        if multi_samples and self.ista2_method == 'v':
            if self.is_swin:
                nW = int(b / real_b)
                real_p2 = nW * p2
            else:
                real_p2 = p2

            if self.rep_tkn_type == 'random':
                # generate the indices of the randomly selected tokens for the 'random' rep_tkn_type
                self._update_random_token_indices(real_p2, device=k.device)

            if self.is_swin:
                # reverse the window partitioned query, key and value tensors
                v = v.reshape(self.num_heads, real_b, real_p2, d)
                # v: (h, real_b, real_p2, d)
                v = self.ista2_v(v, v)
                # v: (h, b, p2, d)
                v = v.reshape(self.num_heads, b, p2, d)
            else:
                # v: (h, b, p2, d)
                v = self.ista2_v(v, v)

        # apply softmax to attn along the last axis
        attn = attn.softmax(dim=-1)
        # apply the dropout layer to attn
        if self.attn_drop_layer is not None:
            attn = self.attn_drop_layer(attn)

        # compute the aggregated tensor
        # x: (h, b, p1, d)
        x = attn @ v
        # x: (h, b, p1, d) -> (b, p1, h, d) -> (b, p1, h*d)
        x = x.permute(1, 2, 0, 3).reshape(b, p1, dim)

        return x
