# Comparing-1D-CNN-and-Concept-Extraction-Model-for-Patient-Phenotyping
## Introduction
This repository contains the code for reimplementing the paper "Comparing deep learning and concept extraction based methods for patient phenotyping". A build-from-scratch 1D CNN language model (LM) used on patient's clinical narratives phenotyping and comparing the LM with concept-extraction-based classification models are presented in this repository.

Patient phenotyping is a classification technique that facilitate the diagnosis of health conditions or highlight the risks categorically of diseases for doctors. A well trained LM can not only save the time of chart review but also extract the peripheral informations, e.g, identifing bill codes from texts.
## Data
In annotations.csv, you can find the health condition annotations as well as IDs for patient visits in MIMIC-III, namely the hospital admission ID, subject ID, and chart time. Due to HIPAA requirements, we cannot provide the text of discharge summary in this repository. Texts are contained in noteevents.csv, which is publicly available on MIMIC-III: https://physionet.org/content/mimiciii/1.4/. But NOTEEVENTS-2.csv with attribute "text" pre-cleansed (provided by members of Team D2) is used in this repo. The NOTEEVENTS-2 dataset with text, subject ID and hospital ID is merged with the annotation dataset with the IDs and conditions (targets) using pyspark. Then the text column is further tokenized using Regex to form a column of list of words with light cleaning. Then Word2Vec is applied to generate word embeddings with customized vector size. Cell 29 of the ipynb file exhibit the top 10 words that are close to "alcohol" in the vectorized space. Not surprisingly, "abuse", "EtOH", and "drinks" are on the list. The preprocess.py helps to load the dataset, w2v, and generate dataloaders. The n-gram based classifation models are coded in the basic_model file using sklearn and the 1D-CNN LM is coded from scratch using torch. Note that the CNN model with trainer and verbose is a reimplementation of the codes in the original paper, which is coded in lua. It is recommended to follow their instruction to install torch, packages, and run their model for practice. https://github.com/sebastianGehrmann/phenotyping.
## Code
Run the cells in CSE6250 Final Peoject1.ipynb. Cell 23 will give the merged dataset. Cell 30 will save the embeddings to w2v.txt. Basic models and 1D CNN are tested from Cell 31. It is recommended to customize n-gram, classification model, embedding vector size for basic model, and regular hypertuning, e.g., batch size, epochs, target name, etc., for CNN LM.

Here are the health conditions: 1: cohort (is the patient frequent flier) 2: Obesity 3: Non Adherence 4: Developmental Delay Retardation 5: Advanced Heart Disease 6: Advanced Lung Disease 7: Schizophrenia and other Psychiatric Disorders 8: Alcohol Abuse 9: Other Substance Abuse 10: Chronic Pain Fibromyalgia 11: Chronic Neurological Dystrophies 12: Advanced Cancer 13: Depression 14: Dementia. Note that some of them are very imbalanced. Positive (1, disease confirmed) is much less than negative for those conditions. Although the weights argument of the CE loss function are assigned, it is highly recommneded to enhance the sampling of the positives by yourself.
## Reference
Gehrmann S, Dernoncourt F, Li Y, et al. Comparing deep learning and concept extraction based methods for patient phenotyping from clinical narratives. PLOS ONE. 2018;13(2):e0192360.

https://github.com/sebastianGehrmann/phenotyping

To access the MIMIC-III data, visit:
https://physionet.org/

Reimplementation of the 1D-CNN LM:
https://github.com/leohsuofnthu/Pytorch-TextCNN
