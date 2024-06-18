# load ViTVQ3D from modules

from modules import ViTVQ3D

# def __init__(self, volume_key: str, volume_size: int, patch_size: int, encoder: OmegaConf, decoder: OmegaConf, quantizer: OmegaConf,
#                  loss: OmegaConf, path: Optional[str] = None, ignore_keys: List[str] = list(), scheduler: Optional[OmegaConf] = None) -> None:

model = ViTVQ3D(volume_key="volume", volume_size=64, patch_size=8, encoder={"dim": 256, "depth": 4, "heads": 4, "mlp_dim": 512, "dropout": 0.1},
                decoder={"dim": 256, "depth": 4, "heads": 4, "mlp_dim": 512, "dropout": 0.1},
                quantizer={"embed_dim": 64, "num_embeddings": 512, "commitment_cost": 0.25, "decay": 0.99},
                loss={"recon_loss": {"type": "mse", "params": {}}, "commitment_loss": {"type": "vq", "params": {}}},
                path=None, ignore_keys=list(), scheduler=None)

