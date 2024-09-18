# monai.networks.nets.SegResNetDS(spatial_dims=3, init_filters=32, in_channels=1, out_channels=2, act='relu', norm='batch', blocks_down=(1, 2, 2, 4), blocks_up=None, dsdepth=1, preprocess=None, upsample_mode='deconv', resolution=None)[source]#

from monai.networks.nets import SegResNetDS

model = SegResNetDS(
    spatial_dims=3,
    init_filters=32,
    in_channels=1,
    out_channels=3,
    act="relu",
    norm="INSTANCE",
    blocks_down=(1, 2, 2, 4, 4, 4),
    # blocks_up=(2, 2, 2, 2),
    dsdepth=4,
    preprocess=None,
    upsample_mode="deconv",
    resolution=None,
)