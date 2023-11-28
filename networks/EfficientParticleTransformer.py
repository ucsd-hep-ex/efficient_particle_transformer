""" Particle Transformer (ParT)

Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
"""
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial
from particle_transformer.networks.multihead_linear_attention import MultiheadLinearAttention

from weaver.utils.logger import _logger
from weaver.nn.model.ParticleTransformer import build_sparse_tensor, trunc_normal_, SequenceTrimmer, Embed, PairEmbed


class LinBlock(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        max_seq_len=128,
        compressed=4,
        ffn_ratio=4,
        dropout=0.1,
        attn_dropout=0.1,
        activation_dropout=0.1,
        add_bias_kv=False,
        activation="gelu",
        scale_fc=True,
        scale_attn=True,
        scale_heads=True,
        scale_resids=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.compressed = compressed
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio

        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiheadLinearAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
            max_seq_len=max_seq_len,
            compressed=compressed,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = (
            nn.Parameter(torch.ones(num_heads), requires_grad=True)
            if scale_heads
            else None
        )
        self.w_resid = (
            nn.Parameter(torch.ones(embed_dim), requires_grad=True)
            if scale_resids
            else None
        )

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if x_cls is not None:
            with torch.no_grad():
                # prepend one element for x_cls: -> (batch, 1+seq_len)
                padding_mask = torch.cat(
                    (torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1
                )
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[
                0
            ]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask)[
                0
            ]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum("tbhd,h->tbdh", x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x


class EfficientParticleTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes=None,
        # network configurations
        embed_dims=[128, 512, 128],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp

        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            compressed=4,
            ffn_ratio=4,
            dropout=0.1,
            attn_dropout=0.1,
            activation_dropout=0.1,
            add_bias_kv=False,
            activation=activation,
            scale_fc=True,
            scale_attn=True,
            scale_heads=True,
            scale_resids=True,
        )

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)
        _logger.info("cfg_block: %s" % str(cfg_block))

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)
        _logger.info("cfg_cls_block: %s" % str(cfg_cls_block))

        self.embed = (
            Embed(input_dim, embed_dims, activation=activation)
            if len(embed_dims) > 0
            else nn.Identity()
        )
        self.blocks = nn.ModuleList([LinBlock(**cfg_block) for _ in range(num_layers)])
        self.cls_blocks = nn.ModuleList(
            [LinBlock(**cfg_cls_block) for _ in range(num_cls_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        if fc_params is not None:
            fcs = []
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(
                    nn.Sequential(
                        nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)
                    )
                )
                in_dim = out_dim
            fcs.append(nn.Linear(in_dim, num_classes))
            self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
        }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None

        with torch.no_grad():
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)

            # fc
            if self.fc is None:
                return x_cls
            output = self.fc(x_cls)
            if self.for_inference:
                output = torch.softmax(output, dim=1)
            # print('output:\n', output)
            return output


class EfficientParticleTransformerTagger(nn.Module):
    def __init__(
        self,
        pf_input_dim,
        sv_input_dim,
        num_classes=None,
        # network configurations
        embed_dims=[128, 512, 128],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[],
        activation="gelu",
        # misc
        trim=True,
        for_inference=False,
        use_amp=False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.use_amp = use_amp

        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)

        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)

        self.part = EfficientParticleTransformer(
            input_dim=embed_dims[-1],
            num_classes=num_classes,
            # network configurations
            embed_dims=[],
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            fc_params=fc_params,
            activation=activation,
            # misc
            trim=False,
            for_inference=for_inference,
            use_amp=use_amp,
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "part.cls_token",
        }

    def forward(
        self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None
    ):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0

        with torch.no_grad():
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            v = torch.cat([pf_v, sv_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask], dim=2)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            x = torch.cat([pf_x, sv_x], dim=0)

            return self.part(x, v, mask)