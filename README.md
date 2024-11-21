# Diabetic Retinopathy Detection with Machine Learning  

## Overview  
This project addresses early diabetic retinopathy (DR) detection challenges using advanced machine learning models. By leveraging CNNs, VGG16, ResNet50, Adaptive Neural Attention Network (ANAN), and other techniques, the solution provides a scalable, automated way to classify DR severity levels. Early detection reduces blindness risks, particularly in rural areas with limited medical access.  

## Key Features  
- **Models Used**:  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - Convolutional Neural Networks (CNNs)  
  - VGG16, ResNet50, Inception V3, EfficientNet b0, Inception ResNet  
  - Adaptive Neural Attention Network (ANAN)  

- **Challenges Addressed**:  
  - Image quality and class imbalance  
  - Reducing overfitting  
  - Optimizing computational costs for real-world use  

- **Results**:  
  - ANAN achieved the highest score of **0.879** (Quadratic Weighted Kappa).  

## Dataset  
Retinal fundus images are labeled by DR severity levels:  
- 0: No DR  
- 1: Mild  
- 2: Moderate  
- 3: Severe  
- 4: Proliferative DR  

The dataset includes **3600 training images** and **2000 testing images**, with preprocessing steps like resizing to 128x128 and data augmentation.  

## Motivation  
India has over 101 million diabetic patients, with 10 million new cases annually. Nearly 90% of these patients are at risk of DR. Early screening can prevent blindness, particularly in underserved regions.  

## Project Workflow  
1. **Preprocessing**: Data cleaning, augmentation, resizing  
2. **Model Training**: Implementation of multiple architectures  
3. **Validation & Testing**: Robust evaluation with K-fold cross-validation  
4. **Evaluation Metrics**: Quadratic Weighted Kappa (QWK)  

## Novelty  
- Combines lightweight, real-time solutions with scalable machine learning techniques.  
- Incorporates attention mechanisms (ANAN) to highlight critical image regions.  

## Future Scope  
This approach can extend to detecting other vision-related diseases, like glaucoma and macular degeneration, enabling comprehensive diagnostics.  

## Contributors  
- **Isha**: VGG16, Inception V3  
- **Ashutosh**: Data Preprocessing  
- **Vaibhav**: Simple CNN, Model Novelty  
- **Siddharth**: Random Forest, Decision Tree  
- **Anett**: ResNet50 Implementation  
- **B. Sri Sairam Gautam**: Adaptive Neural Attention Network  
- **Sai Krupakar**: Model Challenges, Results  
- **Himanshu**: Support Vector Machines  

## How to Run  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yourusername/diabetic-retinopathy-detection.git  
