🚩 Cool Notebook to understand panoptic segmentation inferencing can be used to understand how the whole pipeline works - [Here](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/DETR_panoptic.ipynb)

## We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention ,From where do we take this encoded image?

This encoded image is the output of the encoder.

We know that ResNet Backbone outputs a feature map which is of $2048 * H/32 * W/32$ which is then reduced to 256 channel’s with the help of 1D convolutions(Fully connected layers). This is added with positional encodings and sent to the encoder. The shape of the input is retained in the encoder and the output of the encoder is what is being sent to the **MultiHead Attention Map** along with the box embeddings which is the output of the Decode

We can also see this in the code - 

First an instance of the class `MHAttentionMap` is created in the `__init __` function of `DETRsegm` class

```python
self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
```

Then in the forward function of `DETRsegm`

```python
hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])
bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
```

Here we can see that `hs` and `memory` which are the outputs of the transformer as passed into the `bbox_attention`

Shown below is the code taken from the `forward` of the `Transformer` class. Here we can see that the `memory` is the output of the `encoder` and `hs` is the output of the `decoder`

```python
def forward():
	.
	.
	.

	memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
	hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
	                          pos=pos_embed, query_pos=query_embed)
	return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
```

## We also send dxN Box embeddings to the Multi-Head Attention, we do something here to generate NxMxH/32xW/32 maps  - What do we do here?

d : No. of hidden channels (256) which was the output of the 

N : No of object queries (100)

M : No. of attention heads (8)

H : Original height of the image

W: Original weight of the image

Each box embedding matrix is multiplied with the encoded image (after doing certain operations like normalising and shape changing) in a multi headed attention format. This generates attention maps which are somewhat like the grad cam outputs.

In the code shown below(`forward` function of the `MHAttentionMap` ) we can see the `torch.einsum` is the line where all the magic happens.

```python
def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights
```

The dimension which are not in the output i.e. in `“bqnhw”` are the dimension along which multiplication takes place. Here multiplication takes along the `c` dimension.

```python
	weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)
```

## Then we concatenate these maps with Res5 Block . Where is this coming from?

First note that  `MaskHeadSmallConv` is the class that is responsible for implementing the Convolutional head and upsampling. It was declared in the __ init__ of `DETRsegm` in the following manner

```python
def __init__(self, detr, freeze_detr=False):
    .    
		.
		.
		self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)
# hidden_dim = 256, nheads = 8
```

Then in the forward method of `DETRsegm` - `self.detr.input_proj` which is a 1D convolutional that reduces the number of channels from 2048 to 256 is used to get the input for  `MaskHeadSmallConv`

```python
def forward(self, samples: NestedTensor):
    .
		.
		.
		features, pos = self.detr.backbone(samples)
		..

		bs = features[-1].tensors.shape[0]
		
		src, mask = features[-1].decompose()
		assert mask is not None
		src_proj = self.detr.input_proj(src)
		...

		bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
		
		seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
		outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
		
		out["pred_masks"] = outputs_seg_masks
		return out
```

In the forward of `MaskHeadSmallConv` is where the attention maps from the MultiHead Attention module are first concatenated to the Res5 block the below shown line of code taken from the forward method of `MaskHeadSmallConv` .

```python
x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
```

The activations maps from the 2,3,4 & 5th layers of the ResNet-backbone architecture are stored as Res2, 3 4 & 5. We add these maps with corresponding activation maps and upsample the image until NxH/4xW/4.The upsampling is done with the help of [Feature Pyramid like architectures](https://jonathan-hui.medium.com/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c)

![Imgur](https://imgur.com/dOzGLDe.png)

## Then we perform the above steps - Explain these steps

During the inference we first filter out the detection with a confidence below 0.85. then they compute the per pixel arg-max to determine in which mask each pixel belongs. They then collapse different mask predictions of the same stuff category into in one and filter the empty ones(less than 4 pixels). Essentially the code shown below is what happens in the `PostProcessPanoptic`

```python
scores = out["pred_logits"].softmax(-1)[..., :-1].max(-1)[0]
# threshold the confidence
keep = scores > 0.85

# Plot all the remaining masks
ncols = 5
fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(keep.sum().item() / ncols), figsize=(18, 10))
for line in axs:
    for a in line:
        a.axis('off')
for i, mask in enumerate(out["pred_masks"][keep]):
    print(mask.shape)
    ax = axs[i // ncols, i % ncols]
    ax.imshow(mask, cmap="cividis")
    ax.axis('off')
fig.tight_layout()

out["pred_masks"][keep].shape
>>>torch.Size([17, 200, 300])
```

## Additional Notes

<aside>
💡 The mask head can be trained either jointly, or in a two steps process, where we train DETR for boxes only, then freeze all the weights and train only the mask head for 25 epochs. Experimentally, these two approaches give similar results, we report results using the latter method since it results in a shorter total wall-clock time training.

</aside>

<aside>
💡 The final resolution of the masks has stride 4 and each mask is supervised independently using the DICE/F-1 loss and Focal loss .

</aside>
