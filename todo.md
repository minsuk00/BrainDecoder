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

- save images during iteration
- caching images in dataloader

- pixel level feature extraction 거의 0임.



- 나는 왤케 dataset loading 오래 걸리지
- batch size는 왜 크게 함?



생각 정리
- 원래 SD는 condition shape가 어떻게 되지? -> 똑같이 만들어야되지 않나?
=> c tensor([[[-0.3884,  0.0229, -0.0522,  ..., -0.4899, -0.3066,  0.0675],
         [ 0.0286, -1.3259,  0.3084,  ..., -0.5260,  0.9774,  0.6653],
         [-0.1971,  0.2917,  1.3057,  ..., -1.9834, -1.2367,  1.7625],
         ...,
         [-0.0810,  0.0318,  0.5363,  ..., -1.6231, -0.6092,  1.2734],
         [-0.0743, -0.0030,  0.5208,  ..., -1.5944, -0.6054,  1.2610],
         [-0.0707,  0.0820,  0.5507,  ..., -1.5166, -0.6090,  1.2121]]],
       device='cuda:0')
=> c shape torch.Size([1, 77, 768])


.model.apply_model(x, t, c)


x_recon = self.model(x_noisy, t, **cond)
if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon


outputs.pooler_output -> openai clip output

=> 그러면 transformer가 1~eot token까지의 내용을 압축해서 마지막 token embedding에 넣음?

- 그러면 SD는 [1,77,768] 다 쓰나? 어디선가 필요 없는 부분은 골라내지 않을까?

할것:
<!-- pixel level gan 완성하기 -> loss function, +@? -->
<!-- pixel level gan output ldm input으로 넣기 -->
sample level -> blip 써서 해보기
evaluation metrics 구현

idea:
latent channel 조절? -> what effect?

<!-- Pixel-Level-semantics: -->
<!-- online triplet loss batch size 늘리기? -->

<!-- diff aug (pickel norm 써서 다시 만들기?) 켜기 -->

<!-- training gan without condition first? -->
<!-- how to add condition? -->

<!-- feature extractor 마지막을 sigmoid? -->