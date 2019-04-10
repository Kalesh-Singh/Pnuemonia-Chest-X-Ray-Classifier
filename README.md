# Pneumonia Chest X-ray Classifier #

__Authors:__
+ Kaleshwar Singh (Howard University)
+ Lianne Sanchez (University of Puerto Rico, Mayaguez)
+ Krystalis Rohena (University of Puerto Rico, Mayaguez)


## Table of Contents ##

1. [Introduction](#introduction)
    + [Motivation](#motivation)
    + [Backgroud](#backgroud)
2. [Method](#method)
    + [Finding the Dataset](#finding-the-dataset)
    + [Dataset Analysis](#dataset-analysis)
        + [Feature Engineering](#feature-engineering)
        + [Additional Dataset](#additional-dataset)
3. [Implementation](#implementation)
    + [Choosing an Architecture](#choosing-an-architecture)
    + [Specifications of the Architecture](#specifications-of-the-architecture)
    + [Test Runs](#test-runs)
4. [Results](#results)
5. [Discussion](#discussion)
    + [Visual vs History](#visual-vs-history)
    + [Medical Perspective](#medical-perspective)
    + [Are these Images Any Good for Analysis](#are-these-images-any-good-for-analysis)
6. [Conclusion](#conclusion)
7. [References](#references)

## Introduction ##
### Motivation ###
Pneumonia is the number one leading cause for death in children of five years or younger. Just in 2013, 935,000 children died because of pneumonia. Also, this disease is the main cause of hospitalization in adults in the United States after woman going into labor. Pneumonia is a disease caused by a virus, bacteria or fungi that produces liquid or fluid in the air sacs of the lungs making breathing harder. Usually, patients that have pneumonia suffer from dizziness due to the lack of  oxygen going through their lungs, they also suffer from fever, headaches, loss of appetite and lack of energy. Pneumonia is a clinically diagnosed disease, and the sooner a patient gets diagnosed the sooner they can get treated for such disease. As you may think, time is of the essence in predicting and classifying when a child has pneumonia. Team Amigos wants to be part of the solution, to help to diagnose people especially children that may have this condition, so they can get treatment faster.
### Background ###
Artificial Intelligence algorithms have been in rapid development for the last few years. There are many health related applications using Machine Learning algorithms to predict diseases and to study the human anatomy. In a research by Hosny A et al [10], he comments: “Artificial intelligence (AI) algorithms, particularly deep learning, have demonstrated remarkable progress in image-recognition tasks. Methods ranging from convolutional neural networks to variational autoencoders have found myriad applications in the medical image analysis field, propelling it forward at a rapid pace. Historically, in radiology practice, trained physicians visually assessed medical images for the detection, characterization and monitoring of diseases. AI methods excel at automatically recognizing complex patterns in imaging data and providing quantitative, rather than qualitative, assessments of radiographic characteristics.” This is considered an advancement that can change not just computer science but medical related fields and affect patients directly. Helping doctors detect diseases in their early stages so that they can be treated.

## Method ##
### Finding the Dataset ###
We came across various datasets that we took into consideration for our final project. At first we contemplated working in the Google Draw data set that focused in a multiclass classification of images, but the data set was too large to handle and we felt like we did not have the capacity. Another idea that we consider was a dataset that contained a lot of capacitor that were both good and bad. The idea was to develop a model that would help caitor company determine if the capacitor was good or bad so that they would not lose money on thrown away capacitors. We did not choose this one because when we look at the entries, there was a user in Kaggle that was able to achieve 100% accuracy in the test data. We found that fact to be a little sketchy so we decided to find something else that we could approach instead. After carefully studying all our possibilities we choose the Pneumonia X-ray classification in children under the age of 5 dataset. 

### Dataset Analysis ###
Our data set was downloaded from Kaggle and contained exactly 5,856 images of X-ray from children that were diagnosed with pneumonia and others that were not diagnosed with pneumonia. The data set was divided into three folders: training, validation and test data in which folder the images were classified into folders respectively. 

In our data set analysis we found some common traits in the x ray of patient that had pneumonia and some differences with the x ray of patient that were not diagnosed with pneumonia. The X ray that displayed lungs that suffered from pneumonia will usually have colors like white and gray in one or both areas of the lungs. This is do to the fact that since the air passways will get swollen, less rays will impact the plate in that area and this will reflect as lighter tones. After some more research we actually were able to notice that the passways will also look a little bit blurt out, while in a normal lung they would look like tiny veins. The common thing among the x ray of healthy lungs was that they will look darker in the area and the airpassways would be more clearly distinguished. 

We reviewed more than 50 images of the dataset to get a sense of what the differences between pneumonia and non-pneumonia x-rays looked like. This would prove that the Convolutional Neural Network would be able to spot the differences of the different pictures. Below in Table 1 a few examples of images with clear differences. On the left side, we can observe a clear x ray and on the right side we can observe opacities in the lung area of the right lung.

![Table 1](./readme-imgs/table1.png?raw=true "Table 1: Normal versus pneumonia x rays and their differences")

__Table 1: Normal versus pneumonia x rays and their differences.__

#### Feature Engineering ####
The dataset images were not consistent in dimensions. In order to user VGG19 (which would explain the implementation section) the images had to be 244 x 244px. Therefore we had to do some transformations to the images so they had these dimensions. All the images were scaled so that all of them had the same dimensions.

#### Additional Dataset ####
After a revision of early results, we took a subset of the training images. We choose 240 images of No-pneumonia and 364 of Pneumonia, and try to run our model on that data. We chose clearer examples of non-pneumonia cases and pneumonia cases, even though we are not radiologist or experts in the topic we proceeded to take clear x rays without opacities as non-pneumonia cases, images with opacities as pneumonia cases and the ones that were maybe compromised by the scaling or difficult to read x rays were discarded. We will discuss our findings in the results section. Below in Tables 2, 3 & 4 you can see some examples of clear x rays, x rays with opacities and trashed x rays. Also, we did a review of the validation and test data with the same requirements.

![Table 2](./readme-imgs/table2.png?raw=true "Table 2: Normal clear x rays chosen by Team Amigos.")

__Table 2: Normal clear x rays.__

![Table 3](./readme-imgs/table3.png?raw=true "Table 3: X rays with opacities chosen by Team Amigos.")

__Table 3: X rays with opacities.__

![Table 4](./readme-imgs/table4.png?raw=true "Table 4: X rays compromised by scaling or unclear difference.")

__Table 4: X rays compromised by scaling or unclear difference.__

## Implementation ##
As mentioned before we will be using the method of transfer learning for this project, which is a term in deep learning. Transfer learning establishes that the Convolutional Neural Networks have the capability to learn information about other data sets.  Very few people train CNN from scratch because it is rare to have a dataset of sufficient size [8]. Since we had an original training dataset of close to 3,500 images, we choose to use pre-trained model.

### Choosing an Architecture ###
In Figure 1, we can observe a comparison between different pre-trained models. At the beginning of our project we decided to go with model VGG-19 pre-trained model as observed it has a 70% accuracy which is really good. In later runs of the project we ran VGG-16 which has the same accuracy but less operations.  We noticed that bigger operations number does not mean the model is better. The less operations the model had the less parameters it has which would be better for approaching.

![Figure 1](./readme-imgs/fig1.png?raw=true "Figure 1: Comparison of pre-trained models.")

__Figure 1: Comparison of pre-trained models.__

### Specifications of the Architecture ###
In Figure 2, it can be observed the architecture of VGG-19 which we used for the first run of the project. It is important to understand that we made some additions to the pre-trained model. The following were added to the architecture:
    
+ 5 dense layers in output with ReLU activation.
+ 2 class output layer with softmax activation. 

Important to explain that we used 2 class output layer with softmax activation given that we had to classify an image between two classes either normal lungs or lungs with pneumonia. These two classes are mutually exclusive, since a person cannot have normal lungs and lungs with pneumonia at the same time.

![Figure 2](./readme-imgs/fig2.png?raw=true "Figure 2: VGG-19 pre-trained model architecture.")

__Figure 2: VGG-19 pre-trained model architecture.__

![Figure 3](./readme-imgs/fig3.png?raw=true "Figure 3: Our modified model architecture for early runs.")

__Figure 3: Our modified model architecture for early runs.__

In Figure 4, we can observe the architecture of VGG-16, later on used. We started to use VGG-16 given the advice of Prof. Yoon of testing our model with a known reliable dataset. We tested our model with the cat vs. dogs dataset, and the accuracy was close to 40%. Meaning that there was something wrong with the pipeline of the model that was trained beforehand. By testing the VGG-16 with cats vs dogs we noticed that it got about 85% of accuracy and decided to change our chosen pre-trained model from VGG-19 to VGG-16. The following were added to the architecture:
+ 2 class output layer with softmax activation. Chosen for the reasons explained previously.

![Figure 4](./readme-imgs/fig4.png?raw=true "Figure 4: VGG-16 pre-trained model architecture.")

__Figure 4: VGG-16 pre-trained model architecture.__

![Figure 5](./readme-imgs/fig5.png?raw=true "Figure 5: Our modified model architecture for later runs.")

__Figure 5: Our modified model architecture for later runs.__

### Test Runs ###
For project purposes we tried different configurations and runs. The ones depicted in Table 5 were the most important and the ones with better results. The results of the runs will be discussed in the next sections of the document. In Table 5 the first column is the number of configuration of the model that was tried, the second is the pre-trained model used, the third one is the extra layers used in output of the function. The next column is the dropout in each layer added for example if we added five layers with a dropout of 50% then 50% of the neurons in each layer would be discarded in training. We also specify the optimizer used, the loss, the epochs, batch sizes and SE which stands for Steps per Epochs. Finally, we establish which dataset was used the Original stands for the dataset that was downloaded from Kaggle and passed through a scaling process, the New one stand for the selection of specific cases from the Original. 

![Table 5](./readme-imgs/table5.png?raw=true "Table 5: The generalized attempted runs for the project.")

__Table 5: The generalized attempted runs for the project.__


## Results ##
Since we ran and produced different results with different architectures, in this section we will show plots and confusion matrices for each of the most important runs that we had.

### First Run ###

![Figure 6](./readme-imgs/fig6.png?raw=true "Figure 6: Model Accuracy.")

__Figure 6: Model Accuracy.__

![Figure 7](./readme-imgs/fig7.png?raw=true "Figure 7: Model Loss Curves.")

__Figure 7: Model Loss Curves.__

![Figure 8](./readme-imgs/fig8.png?raw=true "Figure 8: Confusion Matrix.")

__Figure 8: Confusion Matrix.__

### Second Run ###

![Figure 9](./readme-imgs/fig9.png?raw=true "Figure 9: Model Accuracy.")

__Figure 9: Model Accuracy.__

![Figure 10](./readme-imgs/fig10.png?raw=true "Figure 10: Model Loss Curves.")

__Figure 10: Model Loss Curves.__

![Figure 11](./readme-imgs/fig11.png?raw=true "Figure 11: Confusion Matrix.")

__Figure 11: Confusion Matrix.__

### Third Run ###

![Figure 12](./readme-imgs/fig12.png?raw=true "Figure 12: Model Accuracy.")

__Figure 12: Model Accuracy.__

![Figure 13](./readme-imgs/fig13.png?raw=true "Figure 13: Model Loss Curves.")

__Figure 13: Model Loss Curves.__

![Figure 14](./readme-imgs/fig14.png?raw=true "Figure 14: Confusion Matrix.")

__Figure 14: Confusion Matrix.__

### Forth Run ###

![Figure 15](./readme-imgs/fig15.png?raw=true "Figure 15: Model Accuracy.")

__Figure 15: Model Accuracy.__

![Figure 16](./readme-imgs/fig16.png?raw=true "Figure 16: Model Loss Curves.")

__Figure 16: Model Loss Curves.__

![Figure 17](./readme-imgs/fig17.png?raw=true "Figure 17: Confusion Matrix.")

__Figure 17: Confusion Matrix.__

## Discussion ##
In this section our main goal is to explain why we think our model did not perform well enough to be used by medical professionals. There are several elements that could have caused this bad classification behavior. We tried using VGG19 pre-trained model, VGG16 pre-trained model. Also, we tested adding 5 layers to the output of the pre-trained model, tried dropout, tried L2 regularization, adding 1 or 2 layers at the output of the pre-trained model. After all of our intense work trying to get a good model we were unsuccessful. We feel that there are various important reasons that caused this, explained below.
### Visual vs History ###
After reading about the origin of our dataset we found that text mining was used in the descriptions authored by the radiologists to label the data. Meaning, that any sign in the document that indicated that the patient may have had pneumonia was labeled as the patient having pneumonia. A radiologist Oakden Rayner discussed about this anomaly and commented that in order to diagnose a patient with pneumonia a simple x-ray with a description was not enough. It was necessary to have a background of the patient, including conditions and several other details that may lead the doctor to conclude the diagnostic. In which case they x-ray, it’s description, the patient history and a doctor would’ve been the right procedure to label the data instead of using data mining. 

Another problem that Oakden Rayner pointed out, was that of all the pictures in the model 89% of the pictures were labelled as pneumonia. He took it upon himself to label them one by one based on his experience and he found that only 35% of the images in the dataset showed a patient that was potentially suffering from pneumonia. Even Though he clarified that he was not a doctor and that his job was only to write down what he observed in the x-rays, he mentioned that the text mining approach had a bigger margin of error compared to that of a radiologist. The more precise approach would have been the patient head doctor.

![Figure 18](./readme-imgs/fig18.png?raw=true "Figure 18: Text Mining vs Visual.")

__Figure 18: Text Mining vs Visual.__

### Medical Perspective ###
There are several elements that we need to understand in terms of diagnosis, some of these points are taken out from [8].
1. Some diseases or conditions are clinically diagnosed, this means that they are not diagnosed solely based on an image. Some diseases are pneumonia, emphysema and some types of fibrosis [3]. If we look at the process of how pneumonia is diagnosed we see steps. First, the patient gets symptoms and goes to the doctor if doctor suspects pneumonia then an x ray will be ordered and further tests need to happen such as cat scans, pulse oximetry and blood tests. With all of the information given of the tests then the doctor will diagnose pneumonia. This process makes us wonder if a human doctor needs all of this information to make a decision; why would we expect a machine learning to classify a condition of a patient solely based on an image?
2. Secondly, we may argue that x rays are not very accurate when it comes to showing the actual problem. One example of lack of accuracy in x rays is what is called a pneumothorax. A pneumothorax is when there is a big amount of air in the pleural cavity of the lungs. This is deadly given that it can cause the lungs to collapse at any moment. A pneumothorax is really subtle in x rays and they are often missed. As a fact up to 50% of nodules are missed on question, thus compromising in our eyes the fidelity that x rays have from a medical perspective.

### Are these Images Any Good for Analysis ###
What is the model actually learning? There was a popular paper from Zhang et al. a while back which showed that deep learning can fit random labels in training data. Random noise could be a good regularizer, and sometimes it can improve or increase performance. Random noise that improves performance is called label smoothing ot soft labels. Contrary to random noise in our original dataset we had what is called structured noise, this will add a truly different signal that the model will try to learn. In a training set with bad labels, a neural network will treat these as equally valid examples of pneumonia. If the goal is to maximise performance, then structured labeled noise is always a negative.

An example of this, is when dermatologist look at a lesion if they think its malignant include a ruler in the bipsy pictures. By adding the ruler it does not mean that the lesion is malignant, but if you train a model with pictures every picture that has a ruler it will predict that is malignant given the correlation. As  Dr. Novoa from Stanford dermatology emphasizes the algorithm doesn’t know why that correlation makes sense, so it could easily misinterpret a random ruler sighting as grounds to diagnose cancer. The take home message here is that deep learning is really powerful. Indiscriminately so. If you give it complex images with biased labels, it can learn to distinguish between the classes even when the classes are meaningless.



## Conclusion ##
In most of the training runs the model manages to perform well on the training and validation data (scoring  >85%  accuracy on both). However, it does not generalize this to the test set. In these cases, we can infer that the model overfitted and is memorizing the training data. In the case where we attempted to minimize this overfitting by using dropout regularization or by training for less epochs, the model only obtains on average about 54% accuracy on the training data, with similar results on the test set.

Despite good performance on the training and validation dataset, we get really bad classification on the test set, due to the label inaccuracy. Even with our attempt at relabeling, we saw no improvement in performance. However, we should keep in mind that our visual labels were done by eyeballing the images and we still expect it to be far from the correct labels (some source of ground truth, like a panel of radiologists). 

Our findings are consistent with those obtained by Zhang et al. where they found that deep neural networks can perfectly fit random labels in the data set. However, they also showed that the results did not generalize to their test set - effectively demonstrating that structured label noise is detrimental to performance.

In conclusion, the image labels are vastly different than those obtained by visual assessment. This inaccuracy is also present in the test data, meaning that even if the model showed good performance, it could be producing predictions that don’t make medical sense. These problems  mean that the dataset as defined currently, is not fit for training machine learning models, without significant correction to the image labels.


## References ##
[1]https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/home

[2]Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2 http://dx.doi.org/10.17632/rscbjbr9sj.2

[3]https://www.webmd.com/lung/understanding-pneumonia-basics

[4]https://www.mayoclinic.org/diseases-conditions/pneumonia/symptoms-causes/
syc-20354204

[5]https://www.lung.org/lung-health-and-diseases/lung-disease-lookup/pneumonia/sympt
oms-causes-and-risk.html

[6]https://www.nhlbi.nih.gov/health-topics/pneumonia

[7]https://towardsdatascience.com/dont-use-dropout-in-convolutional-networks-81
486c823c16

[8]https://www.quora.com/What-is-meant-be-pre-trained-model-in-CNN-Are-they-already-trained-on-that-particular-classes 

[9]https://lukeoakdenrayner.wordpress.com/2017/12/18/the-chestxray14-dataset-problems/

[10]https://www.ncbi.nlm.nih.gov/pubmed/29777175
