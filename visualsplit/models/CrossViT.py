import math
from copy import deepcopy
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.configuration_utils import PretrainedConfig
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAEAttention,
    ViTMAEIntermediate,
    ViTMAEOutput,
    ViTMAELayer,
    ViTMAEDecoderOutput,
    ViTMAEForPreTraining,
    ViTMAEPatchEmbeddings,
    get_2d_sincos_pos_embed,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from ..utils.extract_features import FeatureExtractor


feature_extractor = FeatureExtractor()


class CrossViTConfig(PretrainedConfig):

    def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=224,
        patch_size=16,
        num_channels=4,
        num_channels_output=3,
        qkv_bias=True,
        decoder_num_attention_heads=16,
        decoder_hidden_size=512,
        decoder_num_hidden_layers=8,
        decoder_intermediate_size=2048,
        mask_ratio=0.75,
        norm_pix_loss=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_channels_output = num_channels_output
        self.qkv_bias = qkv_bias
        self.decoder_num_attention_heads = decoder_num_attention_heads
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_hidden_layers = decoder_num_hidden_layers
        self.decoder_intermediate_size = decoder_intermediate_size
        self.mask_ratio = mask_ratio
        self.norm_pix_loss = norm_pix_loss


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention ViT->ViTMAE
class CrossAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.qkv_bias
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        prompt_embedding,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(prompt_embedding))
        value_layer = self.transpose_for_scores(self.value(prompt_embedding))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        return outputs


class CrossViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViTMAEPatchEmbeddings(config)
        self.num_patches = self.patch_embeddings.num_patches
        # fixed sin-cos embedding
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.hidden_size),
            requires_grad=False,
        )
        self.config = config
        self.initialize_weights()

    def initialize_weights(self):
        # initialize (and freeze) position embeddings by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.position_embeddings.shape[-1],
            int(self.patch_embeddings.num_patches**0.5),
            add_cls_token=True,
        )
        self.position_embeddings.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embeddings like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embeddings.projection.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=self.config.initializer_range)

    def forward(self, pixel_values, noise=None):
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # add position embeddings w/o cls token
        embeddings = embeddings + self.position_embeddings[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.position_embeddings[:, :1, :]
        cls_tokens = cls_token.expand(embeddings.shape[0], -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->ViTMAE
class ViTMAESelfOutput(nn.Module):
    """
    The residual connection is defined in ViTMAELayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->ViTMAE
class CrossViTCrossAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = CrossAttention(config)
        self.output = ViTMAESelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.attention.num_attention_heads,
            self.attention.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(
            heads
        )
        self.attention.all_head_size = (
            self.attention.attention_head_size * self.attention.num_attention_heads
        )
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_embedding: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(
            hidden_states=hidden_states,
            head_mask=head_mask,
            prompt_embedding=prompt_embedding,
            output_attentions=output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class HistogramAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(100, config.hidden_size)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation=config.hidden_act,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer, num_layers=config.num_hidden_layers
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        return self.transformer_encoder(hidden_states)


class CrossViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.histogram_attention = HistogramAttention(config)
        self.attention = ViTMAEAttention(config)
        self.cross_attention = CrossViTCrossAttention(config)
        self.intermediate = ViTMAEIntermediate(config)
        self.output = ViTMAEOutput(config)
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_intermediate = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(
                hidden_states
            ),  # in ViTMAE, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # Intermediate Multi-head Cross-attention
        gray_level_histogram_embedding = self.histogram_attention(cross_attention)

        layer_intermediate = self.layernorm_intermediate(hidden_states)

        cross_attention_outputs = self.cross_attention(
            hidden_states=layer_intermediate,
            prompt_embedding=gray_level_histogram_embedding,
            output_attentions=output_attentions,
        )
        cross_attention_output = cross_attention_outputs[0]

        cross_attention_after_residual = cross_attention_output + hidden_states

        # in ViTMAE, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(cross_attention_after_residual)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, cross_attention_after_residual)

        outputs = (layer_output,) + outputs

        return outputs


class CrossViTModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embeddings = CrossViTEmbeddings(config)
        self.layer = nn.ModuleList(
            [CrossViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def get_input_embeddings(self):
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[tuple]:
        embedding = self.embeddings(hidden_states)
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(embedding, cross_attention)

            hidden_states = layer_outputs[0]

        return hidden_states


class ViTDecoder(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(
            config.hidden_size, config.decoder_hidden_size, bias=True
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size),
            requires_grad=False,
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [
                ViTMAELayer(decoder_config)
                for _ in range(config.decoder_num_hidden_layers)
            ]
        )

        self.decoder_norm = nn.LayerNorm(
            config.decoder_hidden_size, eps=config.layer_norm_eps
        )
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size,
            config.patch_size**2 * config.num_channels_output,
            bias=True,
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # add pos embed
        hidden_states = x + self.decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, head_mask=None, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class CrossViTForPreTraining(ViTMAEForPreTraining):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = CrossViTModel(config)
        self.decoder = ViTDecoder(
            config, num_patches=self.encoder.embeddings.num_patches
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward_loss(self, pixel_values, pred):
        """
        Args:
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
                Pixel values.
            source_edge (`torch.FloatTensor` of shape `(batch_size, 1, height, width)`):
                Edge map.
            source_colour (`torch.FloatTensor` of shape `(batch_size, 2, height, width)`):
                Colour values. AB channels.
            source_gray_level_histogram (`torch.FloatTensor` of shape `(batch_size, 128)`):
                Gray level histogram.
            pred (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Predicted pixel values.
            pred_edge (`torch.FloatTensor` of shape `(batch_size, num_patches, 1)`):
                Predicted edge map.
            pred_colour (`torch.FloatTensor` of shape `(batch_size, num_patches, 2)`):
                Predicted colour values. AB channels.
            pred_gray_level_histogram (`torch.FloatTensor` of shape `(batch_size, num_patches, 128)`):
                Predicted gray level histogram.
        Returns:
            `torch.FloatTensor`: Pixel reconstruction loss.
        """

        loss = torch.nn.functional.mse_loss(pred, pixel_values)
        return loss

    def forward(
        self,
        source_edge,
        source_gray_level_histogram,
        source_segmented_rgb,
        pixel_values: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        edge_and_colour = torch.cat([source_edge, source_segmented_rgb], dim=1)

        outputs = self.encoder(
            edge_and_colour,
            cross_attention=source_gray_level_histogram,
            return_dict=return_dict,
        )

        latent = outputs

        decoder_outputs = self.decoder(latent)

        logits = (
            decoder_outputs.logits
        )  # shape (batch_size, num_patches, patch_size*patch_size*num_channels)

        logits_reshape = self.unpatchify(logits)
        if pixel_values is not None:
            loss = self.forward_loss(pixel_values=pixel_values, pred=logits_reshape)
        else:
            loss = None
        return {
            "loss": loss,
            "latent": latent,
            "logits": logits,
            "logits_reshape": logits_reshape,
        }

    def unpatchify(self, patchified_pixel_values):
        """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
        patch_size, num_channels = (
            self.config.patch_size,
            self.config.num_channels_output,
        )
        num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
        # sanity check
        if num_patches_one_direction**2 != patchified_pixel_values.shape[1]:
            raise ValueError("Make sure that the number of patches can be squared")

        # unpatchify
        batch_size = patchified_pixel_values.shape[0]
        patchified_pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_patches_one_direction,
            num_patches_one_direction,
            patch_size,
            patch_size,
            num_channels,
        )
        patchified_pixel_values = torch.einsum(
            "nhwpqc->nchpwq", patchified_pixel_values
        )
        pixel_values = patchified_pixel_values.reshape(
            batch_size,
            num_channels,
            num_patches_one_direction * patch_size,
            num_patches_one_direction * patch_size,
        )
        return pixel_values


class CrossViT(CrossViTForPreTraining):
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values_shape = pixel_values.shape
        if pixel_values_shape[2] != 224:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values, size=(224, 224), mode="bilinear"
            )
        edge, gray_level, segmented_rgb, ab = feature_extractor(pixel_values)

        edge_and_colour = torch.cat([edge, segmented_rgb], dim=1)

        outputs = self.encoder(
            edge_and_colour,
            cross_attention=gray_level,
            return_dict=True,
        )

        return outputs


class CrossViTForClassification(nn.Module):
    def __init__(self, config, num_label, encoder_pretrained_path=None):
        super().__init__()
        self.config = config
        self.encoder = CrossViT(config)
        self.classifier = nn.Linear(config.hidden_size, num_label)
        if encoder_pretrained_path:
            from safetensors.torch import load_model

            load_model(self.encoder, encoder_pretrained_path)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(pixel_values)
        logits = self.classifier(latent[:, 0, :])
        return logits
