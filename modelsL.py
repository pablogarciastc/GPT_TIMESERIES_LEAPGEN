# ------------------------------------------
# MOMENT backbone registration
# ------------------------------------------
from timm.models.registry import register_model
from moment_transformerL import MomentTransformerL


@register_model
def moment_base(pretrained=False, num_classes=18, **kwargs):
    """
    MOMENT foundation model (large) with prompt support for continual learning.
    """
    return MomentTransformerL(
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs,
    )
