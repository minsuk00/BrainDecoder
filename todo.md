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


## Pixel level semantic
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
- make feature extraction function from brain2image (f)
- implement contrastive learning to train f
- make gan for saliency map