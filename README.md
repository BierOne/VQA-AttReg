# Answer Questions with Right Image Regions: A Visual Attention Regularization Approach

## Introduction
In this work, we devise a novel visual attention regularization approach,
namely AttReg, for better visual grounding in VQA. Specifically, AttReg firstly identifies the image
regions which are essential for question answering yet unexpectedly ignored (i.e., assigned with low attention weights) by the backbone
model. And then a mask-guided learning scheme is leveraged to regularize the visual attention to focus more on these ignored key
regions.
Extensive experiments over three benchmark datasets, i.e., VQA-CP v2, VQA-CP
v1, and VQA v2, have been conducted to evaluate the effectiveness of AttReg. As a by-product, when incorporating AttReg into the
strong baseline LMH, our approach can achieve a new state-of-the-art accuracy.
## Main Results

#### The results on VQA v2
| Model| Validation Accuracy | Test Accuracy 
| --- | -- | -- |
| UpDn | 63.77 | 64.90 |
| UpDn + AttReg | **64.13** | **65.25** |

| Model| Validation Accuracy | Test Accuracy 
| --- | -- | -- |
| LMH | 62.40 | -- |
| LMH + AttReg | **62.74** | -- |

#### The results on VQA-CP Test
| Model| VQA-CP v1 Test Accuracy | VQA-CP v2 Test Accuracy
| --- | -- | -- |
| UpDn | 38.88 | 40.09 |
| UpDn + AttReg | **47.66** | **46.85** |

| Model| Validation Accuracy | Test Accuracy 
| --- | -- | -- |
| LMH | 55.73 | 52.99 |
| LMH + AttReg | **62.25** | **59.92** |

Note:
- the training time is about 4 min per epoch (Titan Xp).
- The accuracy was calculated using the [VQA evaluation metric](http://www.visualqa.org/evaluation.html).
- This implementation follows my re-implementation of the VQA-UpDn model 
(refer to "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering"
, https://arxiv.org/abs/1707.07998), which is available at https://github.com/BierOne/bottom-up-attention-vqa.


## Prerequisites
- python3.7 is suggested, though python2 is also compatible
- pytorch (v1.4 is suggested)
- tqdm, h5py, pickle, numpy


## Preprocessing
1. Extract pre-trained image features. (available at )
    ```
    python preprocess-image-rcnn.py
    ```
2. Preprocess questions and answers.
    ```
    python tools/compute_softscore.py
    python tools/create_dictionary.py
    ```
3. Extract key image objects using QA explanation.
    ```
    python tools/create_explanation.py --exp qa
    python tools/preprocess-hint.py --exp qa
    ```

## Training
1. Pre-train the baseline model. (Pls edit the utilities/config.py to change the
baseline model)
    ```
    python main.py --output baseline --pattern baseline --gpu 0
    ```
   
2. Fine-tune the pre-trained model with AttReg.
    
    - if using the UpDn as the baseline:
        ```
        python main.py --output baseline --gpu 0 --pattern finetune --resume --lr 2e-5 --lamda 5
        ```
    - if using the LMH as the baseline:
        ```
        python main.py --output baseline --gpu 0 --pattern finetune --resume --lr 2e-4 --lamda 0.5
        ```
## Citation
    @article{VQA-AttReg,
      title={Answer Questions with Right Image Regions: A Visual Attention Regularization Approach},
      author={Yibing Liu, Yangyang Guo, Jianhua Yin, Xuemeng Song, Weifeng Liu, Liqiang Nie},
      journal={arXiv preprint arXiv:2102.01916},
      year={2021}
    }
