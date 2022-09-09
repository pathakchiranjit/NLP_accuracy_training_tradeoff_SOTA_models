### **News Article Classification Task using SOTA models**

<img src="https://github.com/pathakchiranjit/NLP_accuracy_training_tradeoff_SOTA_models/blob/main/pics/title.jpg?raw=true" align='center'><br/>

**Data Source:**

[Popular News classification dataset](https://www.kaggle.com/rmisra/news-category-dataset)

**Context**

This dataset contains around 200k news headlines from the year 2012 to 2018 obtained from [HuffPost](https://www.huffpost.com/). 

The model trained on this dataset could be used to identify tags for untracked news articles or to identify the type of language used in different news articles.

**RESULT**
<img src="https://github.com/pathakchiranjit/NLP_accuracy_training_tradeoff_SOTA_models/blob/main/pics/model_eval.png?raw=true" align='center'><br/>

**Brief History of NLP world**

While there are plenty of articles on the internet about the **state of the art BERT** algorithm and its applications , its probably also important to understand the journey of the Natural Language Processing science from a rather primitive form to what it is now. There has been fair number of milestones in the way in form of novel techniques ranging from embedding techniques, improved language models, neural architectures and so on.

<img src="https://github.com/pathakchiranjit/NLP_accuracy_training_tradeoff_SOTA_models/blob/main/pics/NLP_models.png?raw=true" align='center'><br/>


**BERT and its natives**

Google’s BERT and recent transformer-based methods have taken the NLP landscape by a storm, outperforming the state-of-the-art on several tasks.

[BERT](https://arxiv.org/abs/1810.04805) is a bi-directional transformer for pre-training over a lot of unlabeled textual data to learn a language representation that can be used to fine-tune for specific machine learning tasks.

[XLNet](https://arxiv.org/abs/1906.08237) is a large bidirectional transformer that uses improved training methodology, larger data and more computational power to achieve better than BERT prediction metrics on 20 language tasks.

[RoBERTa](https://arxiv.org/pdf/1907.11692.pdf) Introduced at Facebook, Robustly optimized BERT approach RoBERTa, is a retraining of BERT with improved training methodology, 1000% more data and compute power.

[DistilBERT](https://arxiv.org/pdf/1910.01108.pdf) learns a distilled (approximate) version of BERT, retaining 97% performance but using only half the number of parameters (paper). Specifically, it does not has token-type embeddings, pooler and retains only half of the layers from Google’s BERT. DistilBERT uses a technique called distillation, which approximates the Google’s BERT, i.e. the large neural network by a smaller one.

<img src="https://github.com/pathakchiranjit/NLP_accuracy_training_tradeoff_SOTA_models/blob/main/pics/model_descriptions.png?raw=true" align='center'><br/>


## Table of Contents

1. [Objective: Problem Statement](#section1)<br>
2. [Tools : Importing Packages, Libraries & Defining Functions:](#section2)<br>
  - 2.1 [Import Packages and Libraries:](#section201)<br>
  - 2.2 [Defining Functions :](#section202)<br>
    - 2.2.1 [missing_data : To find missing values in dataset](#section2021)<br>
    - 2.2.2 [stratified_sample : to sample from dataset with proper proportion of all classes/categories](#section2022)<br>
    - 2.2.3 [creat_bert_input_features : Preparing dataset features for BERT to perform](#section2023)<br>
    - 2.2.4 [creat_dist_bert_input_features : Preparing dataset features for DistilBERT to perform](#section2024)<br>
    - 2.2.5 [remove_patterns and preprocess_text : for preprocessing the text data and clean](#section2025)<br>
3. [Collecting & Loading Data](#section3)<br>
  - 3.1 [ Import dataset from G-Drive](#section301)<br>
4. [Data Preprocessing](#section4)<br>
  - 4.1 [Merging similar topics together](#section401)<br>
  - 4.2 [Stratified Sample of dataframe for model building](#section402)<br>
  - 4.3 [Data Cleaning and Preprocessing](#section403)<br>
  - 4.4 [Preparing the Train, Validation and Test data](#section404)<br>
5. [Model Building and Testing](#section5)<br>
  - 5.1 [Using BERT pretrained model : as embedding layer followed by multiclass classification using shallow Neural Network](#section501)<br>
  - 5.2 [Using DistilBERT pretrained model : as embedding layer followed by multiclass classification using shallow Neural Network](#section502)<br>
6. [Conclusion](#section6)<br>
7. [Actionable Insights:](#section7)
8. [Limitation of the study:](#section8)

