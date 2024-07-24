## Baseline model

We need to train basic UNet models for PET to CT study. We will use the following configurations:

- Depth from 3 to 5
- Number of filters from 16 to 64
- Kernel combinations incluind:
    - 3x3
    - 3x3 + 3x3
    - 3x3 + 3x3 + 3x3
    - 1x1
- SE block
    - True / False
    - Reduction Ratio: 2, 4, 8

## For PET feature extractions:

We aims to let the PET encoder can extract similar features as CT encoder

- Structure of PET encoder: same, twice, triple number of filters as CT encoder
- Input images
    - Raw PET
    - Raw PET + low-pass PET(Gaussian)
    - Raw PET + synCT from UNet above
    - synCT from UNet above
- Loss function
    - MSE for encoder outputs = MSE(z_CT, z_PET)
    - VQ loss: L_VQ = MSE(sg(z_PET), e_CT) + beta * MSE(z_PET, sg(e_CT))
    - InfoNCE loss: L_InfoNCE = -log(exp(sim(z_PET, e_CT)) / sum(exp(sim(z_PET, e_CT))))


## Literature review

- doi:10.1088/1361-6560/ab4eb7