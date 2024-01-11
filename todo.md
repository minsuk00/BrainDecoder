# NeuroImagen

NeuroImagen
- dataset
- model config
- generate image
- pixel level semantic
- sample level semantic


## dataset
### basic info
11964 segments total (6 sets of train-val-test)
999 segments (single) (1 set of train-val-test)
1 segment => 128channels x 440 samples

### todo
- visualize
- write code for dataset and dataloader


## Pixel level semantics
### info
Pixel-level semantics Extractor Mp
- contrastive learning to learn feature extraction function 
    - (bring together embeddings of eeg when people get similar visual stimulus i.e. seeing images from same class)
    - minimize distance between eeg with same label, maximize if different
    ![Alt text](temp/image.png)
    - f-theta is feature extraction function that maps EEG to feature space
        ![Alt text](temp/image-1.png)
    - Beta to avoid the compression of data representations into a small cluster by the feature extraction network
- Make saliency map
    - use GAN
    - use SSIM loss to enforce accuracy of generated saliency map
    ![Alt text](temp/image-4.png)
    ![Alt text](temp/image-2.png)
    ![Alt text](temp/image-3.png)
    
### todo
- get good accuracy from contrastive loss
- contrastive loss to cgan to get reasonable images
- proceed

## Sample level semantics
### info
- goal: derive some coarse-grained information such as the category of the main object
- relatively easier to be aligned
- extractor M<sub>s</sub>
    1. image caption ground truth from additional annotation model (2 approaches)
        1. use class name of image as caption (an image of ~) (because imagenet class anyways)
        2. use BLIP. (with default pretrained parameters) This is more detailed than class name
    2. align EEG embeddings to generated image caption
        1. get latent embeddings of image caption using CLIP (h_hat)
        2. align the output of EEG sample level encoder (h) with loss L2-norm squared of h-h_hat

## Combine multi level semantics with Diffusion Model
- used the latent diffusion model to perform image-to-image reconstructing with the guidance of conditional text prompt embeddings
1. reconstruct pixel-level semantics G(fθ(x)) and resize it to the resolution of observed visual stimuli
2. G(fθ(x)) is then processed by the encoder E<sub>ldm</sub> of autoencoder from the latent diffusion model and added noise through the diffusion process
3. integrate sample-level semantics h<sub>clip</sub> as the cross-attention input of the U-Net to guide the denoising process
4. project the output of the denoising process to image space with Dldm and finally reconstruct the high-quality image


## Evaluation matrix
..?


- cond stage model get learned conditioning return c, re_latent??

- understand dreamdiffusion's approach to encode eeg
- try classifier?
- read different approaches to encode eeg


참고:
- EEG2IMAGE for pixel level generation




- semi hard triplet mining