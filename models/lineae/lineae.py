import torch
from torch import nn

from .backbones import build_backbone
from .hybrid_encoder import build_hybrid_encoder
from .decoder import build_decoder
from .linea_utils import select_top_line_predictions

from ..registry import MODULE_BUILD_FUNCS


class LINEAE(nn.Module):
    """ This is the Cross-Attention Detector module that performs object detection """
    def __init__(self,
        backbone,
        encoder,
        decoder,
        return_distill_features=False,
        distill_projection_channels=None,
        ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.return_distill_features = bool(return_distill_features)
        self.distill_feature_projections = None
        if distill_projection_channels is not None:
            if len(distill_projection_channels) != 3:
                raise ValueError('distillation projection channels must have three entries')
            self.distill_feature_projections = nn.ModuleList([
                nn.Conv2d(source_channels, target_channels, kernel_size=1)
                for source_channels, target_channels in zip(
                    backbone.out_channels, distill_projection_channels, strict=True
                )
            ])
        
    def forward(self, samples, targets=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features = self.backbone(samples)

        distill_features = None
        if self.return_distill_features:
            if self.distill_feature_projections is None:
                distill_features = features
            else:
                distill_features = [
                    projection(feature)
                    for projection, feature in zip(
                        self.distill_feature_projections, features, strict=True
                    )
                ]

        features = self.encoder(features)

        out = self.decoder(features, targets)
        if distill_features is not None:
            out['distill_features'] = distill_features

        return out

    def deploy(self, ):
        self.eval()
        # Feature projections exist only to train heterogeneous students against
        # the XL teacher. Keep them for strict checkpoint loading, then remove
        # them from the inference graph and parameter count.
        self.return_distill_features = False
        self.distill_feature_projections = None
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select) -> None:
        super().__init__()
        self.num_select = int(num_select)
        self.deploy_mode = False

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']
        out_logits, out_line = select_top_line_predictions(
            out_logits, out_line, self.num_select
        )

        scores = out_logits[..., 0].sigmoid()

        # convert to [x0, y0, x1, y1] format
        lines = out_line * target_sizes.repeat(1, 2).unsqueeze(1)

        if self.deploy_mode:
            return lines, scores

        results = [
            {'lines': predicted_line, 'scores': score}
            for score, predicted_line in zip(scores, lines)
        ]

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self

@MODULE_BUILD_FUNCS.registe_with_name(module_name='LINEAE')
def build_lineae(args):
    backbone = build_backbone(args)
    encoder = build_hybrid_encoder(args)
    decoder = build_decoder(args)

    feature_weight = float(getattr(args, 'distill_feature_weight', 0.0))
    return_distill_features = bool(
        getattr(args, 'return_distill_features', False) or feature_weight > 0
    )
    projection_channels = None
    if feature_weight > 0:
        projection_channels = tuple(getattr(args, 'distill_teacher_feature_channels'))

    model = LINEAE(
        backbone,
        encoder,
        decoder,
        return_distill_features=return_distill_features,
        distill_projection_channels=projection_channels,
    )

    num_select = int(getattr(args, 'num_select', args.num_queries))
    if not 0 < num_select <= args.num_queries:
        raise ValueError('num_select must be in [1, num_queries]')
    postprocessors = PostProcess(num_select)

    return model, postprocessors
