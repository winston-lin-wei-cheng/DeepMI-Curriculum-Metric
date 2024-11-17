# DeepMI-Curriculum-Metric for SER
This is an PyTorch implementation of the *DeepMI-Curriculum-Metric* for attribute-based (i.e., arousal, dominance and valence) speech emotion recognition (SER) framework in the [paper](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6570655). 

![The DeepMI-metric extraction Procedure](/images/DeepMI-Resnet.png)



# Suggested Environment and Requirements
1. Python 3.6+
2. Ubuntu 18.04
3. CUDA 10.0+
4. pytorch version 1.4.0
5. faiss version 1.6.0
6. The scipy, numpy, pandas...etc common used python packages
7. The MSP-Podcast v1.8 (or any other version) labeled and unlabeled sets (request to download from [UTD-MSP lab website](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html))
8. Using the IS13_ComParE.func config to extract 6,373-dim sentence-level functional representations (i.e., handcrafted acoustic features) by [OpenSmile toolkit](https://www.audeering.com/opensmile/). For details of this feature set, please refer to [INTERSPEECH2013 Computational Paralinguistics Challenge](https://www.isca-archive.org/interspeech_2013/schuller13_interspeech.pdf)



# How to run
Remember to firstly extract the 6,373-dim functional features for the entire corpus, we will use the *'labels_concensus.csv'* provided by the corpus as the default input label setting.

**Step1. Building/PreTrained the SSL-DeepEmoCluster model with DenseResNet structure**  
1. the function **getPaths_unlabel_Podcast** in the **utils.py** can adjust the number of unlabeled smaples that have involoved during the semi-supervised training process (i.e., control the desired unlabeled set size, the default size is 90K)
2. change data & label dirs in **norm_para.py**, then run it. this will create a *'NormTerm'* folder that contains label and feature normalization (z-norm) parameters based on the training set. We also provide the parameters of the v1.8 corpus in the *'NormTerm'* folder.
3. change data & label dirs in **training_DeepEmoCluster.py**, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion attributes (Act, Dom or Val)
   * -nc: number of the Kmeans clusters
   * run in the terminal
```
python training_DeepEmoCluster.py -ep 200 -batch 512 -emo Val -nc 50
```
4. for more details about the DeepEmoCluster framework, please refer to [ICASSP2021 paper](https://ieeexplore.ieee.org/abstract/document/9414035)


**Step2. Extract DeepMI by the trained DeepEmoCluster model**
1. change data & label dirs in **extract_DeepMI.py**, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion attributes (Act, Dom or Val)
   * -nc: number of the Kmeans clusters
   * run in the terminal
```
python extract_DeepMI.py -ep 200 -batch 512 -emo Val -nc 50
```
2. args of the **extract_DeepMI.py** need to correspond to the trained models from **training_DeepEmoCluster.py** 
3. this will output a *'curriculum_metric'* folder that contains DeepMI values for the training set to perform curriculum learning (saving in JSON format)


**Step3. Building/Testing curriculum SER model based on DeepMI**
1. change data & label dirs in **training_Curriculum.py** and **testing_Curriculum.py**, the running args are,
   * -ep: number of epochs
   * -batch: batch size for training
   * -emo: emotion attributes (Act, Dom or Val)
   * run in the terminal
```
python training_Curriculum.py -ep 70 -batch 512 -emo Val
python testing_Curriculum.py -ep 70 -batch 512 -emo Val
```
2. the provided testing code is only for the test set of the MSP-Podcast corpus, and it will directly return a CCC performance printing out to the console window



# Pre-trained models & DeepMI metrics
We provide the trained SSL-DeepEmoCluster model weights based on **version 1.8** of the MSP-Podcast under the *'trained_SSLmodels_v1.8'* folder. Users can directly apply these models to extract DeepMI on your own corpus or tasks (remember to get normalization parameters for z-norm of acoustic features and there might have severe mismatch conditions if don't finetune the model!). We also provide the extracted DeepMI metric for the training set of MSP-Podcast v1.8 under the *'curriculum_metric_v1.8'* folder, for users that interesting in analyze important properties of the high information training samples. 

The models training by DeepMI curriculum learning approach are very stable (regardless of different init weights), where the matched (within corpus evaluation, the MSP-Podcast Test set) and mismatched (i.e., cross corpora evaluation, the IEMOCAP and the MSP-IMPROV datasets) CCC performances are listed in the Table below. 

| Testing Corpus       | Act              | Val              | Dom              |
|:--------------------:|:----------------:|:----------------:|:----------------:|
| MSP-Podcast Test set | 0.626            | 0.304            | 0.553            |
| IEMOCAP              | 0.442            | 0.204            | 0.292            |
| MSP-IMPROV           | 0.543            | 0.331            | 0.386            |



# Reference
If you use this code, please cite the following paper:

Wei-Cheng Lin, Kusha Sridhar and Carlos Busso, "An Interpretable Deep Mutual Information Curriculum Metric for A Robust and Generalized Speech Emotion Recognition System"

```
@article{LinDeepMI_2024,
  author = {W.-C. Lin and K. Sridhar and C. Busso},
  title = {An Interpretable Deep Mutual Information Curriculum Metric for A Robust and Generalized Speech Emotion Recognition System},
  journal = {IEEE/ACM Transactions on Audio, Speech and Language Processing},
  volume = {},
  number = {},
  year = {2024},
  pages = {},
  month = {},
  doi={},
}
```
