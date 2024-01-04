# Comparing-1D-CNN-and-Concept-Extraction-Model-for-Patient-Phenotyping
## Introduction
This repository contains the code for reimplementing the paper "Comparing deep learning and concept extraction based methods for patient phenotyping". A build-from-scratch 1D CNN language model (LM) used on patient's clinical narratives phenotyping and comparing the LM with concept-extraction-based classification models are presented in this repository.

Patient phenotyping is a classification technique that facilitate the diagnosis of health conditions or highlight the risks categorically of diseases for doctors. A well trained LM can not only save the time of chart review but also extract the peripheral informations, e.g, identifing bill codes from texts.
## Data
In annotations.csv, you can find the health condition annotations as well as IDs for patient visits in MIMIC-III, namely the hospital admission ID, subject ID, and chart time. Due to HIPAA requirements, we cannot provide the text of discharge summary in this repository. Texts are contained in noteevents.csv, which is publicly available on MIMIC-III: https://physionet.org/content/mimiciii/1.4/. But NOTEEVENTS-2.csv with attribute "text" pre-cleansed (provided by members of Team D2) is used in this repo. The NOTEEVENTS-2 dataset with text, subject ID and hospital ID is merged with the annotation dataset with the IDs and conditions (targets) using pyspark. Then the text column is further tokenized using Regex to form a column of list of words with light cleaning. Then Word2Vec is applied to generate word embeddings with customized vector size. Cell 29 of the ipynb file exhibit the top 10 words that are close to "alcohol" in the vectorized space. Not surprisingly, "abuse", "EtOH", and "drinks" are on the list. The preprocess.py helps to load the dataset, w2v, and generate dataloaders. The n-gram based classifation models are coded in the basic_model file using sklearn and the 1D-CNN LM is coded from scratch using torch. Note that the CNN model with trainer and verbose is a reimplementation of the codes in the original paper, which is coded in lua. It is recommended to follow their instruction to install torch, packages, and run their model for practice. https://github.com/sebastianGehrmann/phenotyping.

