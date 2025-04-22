import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.ops import misc as misc_nn_ops
# in https://blog.csdn.net/oYeZhou/article/details/116664399

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def freeze_layers(resnet, trainable_layers=3):
    # trainable_layers 為 0~4，決定要訓練幾層：layer4、layer3、layer2、layer1、conv1
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]

    for name, parameter in resnet.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
        reduction=16
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(planes * self.expansion, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def se_resnet50_fpn(pretrained=False, weights = "DEFAULT",  norm_layer=None, trainable_layers=3):
    # Step 1: 取得原始 ResNet 模型 
    weights = ResNet50_Weights.verify(weights)
    resnet = ResNet(block=Bottleneck, layers=[3, 4, 6, 3], norm_layer=norm_layer)
    resnet.load_state_dict(weights.get_state_dict(progress=True, check_hash=True), strict=False)

    # if pretrained:
    #     # 如果你有 ImageNet pretrained weights，這裡可以手動載入
    #     state_dict = torch.hub.load_state_dict_from_url(
    #         "https://download.pytorch.org/models/resnet50-0676ba61.pth", progress=True
    #     )
    #     resnet.load_state_dict(state_dict, strict=False)

    # freeze_layers(resnet, trainable_layers=trainable_layers)
    # # Step 2: 指定哪些 layers 要當作 FPN 輸出
    # return_layers = {
    #     "layer1": "0",  # usually res2
    #     "layer2": "1",  # res3
    #     "layer3": "2",  # res4
    #     "layer4": "3",  # res5
    # }

    # in_channels_stage2 = 256
    # in_channels_list = [
    #     in_channels_stage2,
    #     in_channels_stage2 * 2,
    #     in_channels_stage2 * 4,
    #     in_channels_stage2 * 8,
    # ]
    # out_channels = 256

    # body = IntermediateLayerGetter(resnet, return_layers=return_layers)

    # # Step 3: 包成 FPN backbone
    # backbone = BackboneWithFPN(
    #     body,
    #     in_channels_list=in_channels_list,
    #     out_channels=out_channels,
    #     return_layers = return_layers
    # )

    return resnet

def resnet_fpn_backbone_se_version(
    *,
    weights,
    norm_layer = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers = None,
    extra_blocks = None,
) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> import torch
        >>> from torchvision.models import ResNet50_Weights
        >>> from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        >>> backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Args:
        backbone_name (string): resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        weights (WeightsEnum, optional): The pretrained weights for the model
        norm_layer (callable): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default, a ``LastLevelMaxPool`` is used.
    """
    # backbone = resnet.__dict__[backbone_name](weights=weights, norm_layer=norm_layer)
    backbone = se_resnet50_fpn(pretrained=True, weights=weights, norm_layer=norm_layer)
    return _resnet_fpn_extractor(backbone, trainable_layers, returned_layers, extra_blocks)

## 接者處理 se_resnet50_fpn的邏輯確認

def _resnet_fpn_extractor(
    backbone,
    trainable_layers: int,
    returned_layers = None,
    extra_blocks = None,
    norm_layer = None,
) -> BackboneWithFPN:

    # select layers that won't be frozen
    if trainable_layers < 0 or trainable_layers > 5:
        raise ValueError(f"Trainable layers should be in the range [0,5], got {trainable_layers}")
    layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
    if trainable_layers == 5:
        layers_to_train.append("bn1")
    for name, parameter in backbone.named_parameters():
        if all([not name.startswith(layer) for layer in layers_to_train]):
            parameter.requires_grad_(False)

    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )

class Model(nn.Module):
    def __init__(self, training=True):
        super().__init__()
        self.training = training
        # 使用自定義 backbone (含 SEBlock 的 ResNet50)
        backbone = resnet_fpn_backbone_se_version(weights="DEFAULT", trainable_layers=3)  # 不載入不相容的預訓練權重

        self.model = MaskRCNN(backbone, num_classes=5)

        #***********************************************************************************
        # 0.32 USE 50 epoch and se block 0.27 no better
        # backbone = resnet_fpn_backbone('resnet50',weights="DEFAULT", trainable_layers=3)

        # num_classes = 5
        # self.model = MaskRCNN(backbone, num_classes=num_classes)
        #***********************************************************************************

        # # 替換 mask predictor
        # in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        # hidden_layer = 256
        # self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(
        #     in_features_mask, hidden_layer, num_classes
        # )

    def forward(self, x, target=None):
        if target is not None:
            output = self.model(x, target)
        else:
            output = self.model(x)
        return output


def get_model(training=True):
    """Return Model with faster rcnn"""
    return Model(training=training)
