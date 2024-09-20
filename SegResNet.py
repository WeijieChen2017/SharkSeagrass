# # monai.networks.nets.SegResNetDS(spatial_dims=3, init_filters=32, in_channels=1, out_channels=2, act='relu', norm='batch', blocks_down=(1, 2, 2, 4), blocks_up=None, dsdepth=1, preprocess=None, upsample_mode='deconv', resolution=None)[source]#

# from monai.networks.nets import SegResNetDS

# model = SegResNetDS(
#     spatial_dims=3,
#     init_filters=32,
#     in_channels=1,
#     out_channels=3,
#     act="relu",
#     norm="INSTANCE",
#     blocks_down=(1, 2, 2, 4, 4, 4),
#     # blocks_up=(2, 2, 2, 2),
#     dsdepth=4,
#     preprocess=None,
#     upsample_mode="deconv",
#     resolution=None,
# )

# input_res = (192, 192, 128)


SegResNetDS(
  (encoder): SegResEncoder(
    (conv_init): Conv3d(1, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
    (layers): ModuleList(
      (0): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (1): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (1): SegResBlock(
            (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (2): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (1): SegResBlock(
            (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (3): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (1): SegResBlock(
            (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (2): SegResBlock(
            (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (3): SegResBlock(
            (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (4): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (1): SegResBlock(
            (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (2): SegResBlock(
            (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (3): SegResBlock(
            (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Conv3d(512, 1024, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
      )
      (5): ModuleDict(
        (blocks): Sequential(
          (0): SegResBlock(
            (norm1): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (1): SegResBlock(
            (norm1): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (2): SegResBlock(
            (norm1): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
          (3): SegResBlock(
            (norm1): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act1): ReLU(inplace=True)
            (conv1): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
            (norm2): InstanceNorm3d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
            (act2): ReLU(inplace=True)
            (conv2): Conv3d(1024, 1024, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          )
        )
        (downsample): Identity()
      )
    )
  )
  (up_layers): ModuleList(
    (0): ModuleDict(
      (upsample): UpSample(
        (deconv): ConvTranspose3d(1024, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False)
      )
      (blocks): Sequential(
        (0): SegResBlock(
          (norm1): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act1): ReLU(inplace=True)
          (conv1): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (norm2): InstanceNorm3d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act2): ReLU(inplace=True)
          (conv2): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (head): Identity()
    )
    (1): ModuleDict(
      (upsample): UpSample(
        (deconv): ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False)
      )
      (blocks): Sequential(
        (0): SegResBlock(
          (norm1): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act1): ReLU(inplace=True)
          (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (norm2): InstanceNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act2): ReLU(inplace=True)
          (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (head): Conv3d(256, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
    (2): ModuleDict(
      (upsample): UpSample(
        (deconv): ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False)
      )
      (blocks): Sequential(
        (0): SegResBlock(
          (norm1): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act1): ReLU(inplace=True)
          (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (norm2): InstanceNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act2): ReLU(inplace=True)
          (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (head): Conv3d(128, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
    (3): ModuleDict(
      (upsample): UpSample(
        (deconv): ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False)
      )
      (blocks): Sequential(
        (0): SegResBlock(
          (norm1): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act1): ReLU(inplace=True)
          (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (norm2): InstanceNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act2): ReLU(inplace=True)
          (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (head): Conv3d(64, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
    (4): ModuleDict(
      (upsample): UpSample(
        (deconv): ConvTranspose3d(64, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), output_padding=(1, 1, 1), bias=False)
      )
      (blocks): Sequential(
        (0): SegResBlock(
          (norm1): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act1): ReLU(inplace=True)
          (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (norm2): InstanceNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
          (act2): ReLU(inplace=True)
          (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        )
      )
      (head): Conv3d(32, 3, kernel_size=(1, 1, 1), stride=(1, 1, 1))
    )
  )
)