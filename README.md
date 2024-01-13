# Medical Complaint Prediction API
I created a public API that interprets users' description of their medical situations and predicts potential chief complaint. The primary goal is to assist users of the Quin app in understanding their medical symptoms and guiding them on their next steps. Key highlights of the project include:
- Data Analysis and Preprocessing: Analyzed and cleaned a dataset comprising 3,974 user inputs and 75 labels, identifying class imbalance and framing the problem as a multi-label text classification task.
- Model Development and Experimentation: Conducted experiments with various machine learning algorithms and spaCy's NLP models to identify the most effective solution for our specific use case.
- API Development (Python script) and Containerizing: Developed a user-friendly API using FastAPI (includes automated testing via pytest), which was then containerized using Docker
- Cloud Deployment: Deployed the containerized API on Azure Container Instances

## Code and Resources Used
**Python Version**: 3.8 \
**Primary Development Environment**: \
    - Google Colab for data analysis, preprocessing, and model development. \
    - Local Python environment for API development and testing. \
**Packages**: os, re, pandas, numpy, matplotlib, seaborn, sklearn, skmultilearn, joblib, spacy, random, fastapi, httpx, pytest, uvicorn \
**Data Sources**: user_inputs.csv, labels.csv

## How to Use 
The API is deployed at 20.113.106.255. You can send POST requests to get predictions: 
```
# powershell
$uri = 'http://20.113.106.255/predict'
$headers = @{
    'accept' = 'application/json'
    'Content-Type' = 'application/json'
}
$body = '{"text": "ik heb echt de ergste hoofdpijn van mijn leven wat is er aan de hand"}'
$response = Invoke-WebRequest -Uri $uri -Method POST -Headers $headers -Body $body
$response.Content
```

```
# curl
curl -X 'POST' `
  'http://20.113.106.255/predict' `
  -H 'accept: application/json' `
  -H 'Content-Type: application/json' `
  -d '{"text": "ik heb echt de ergste hoofdpijn van mijn leven wat is er aan de hand"}'
```

## Data Cleaning and Preprocessing
The dataset consists of 3974 user inputs with 75 potential labels. I cleaned to remove unnecessary columns (i.e. no complaint was not used at all) and preprocessed for text analysis, including converting to lowercase and removing special characters.

## EDA
I looked at the class distribution and the nature of the multi-label classification problem. This included analyzing label frequencies and the correlation between different labels. From our EDA, we found class imbalance, slight multi-label (most user inputs, 84%, are associated with only 1 label), and low correlations between labels.

## Model Building
I tried two main algorithms, where in each subset I also tried different models:
1. **ML Algorithms**
    - **Purpose**: Established as a baseline to gauge performance.
    - **Data Preparation**: I used TF-IDF for text vectorization and `iterative_train_test_split` for a stratified split, to preserve the distribution of labels in both training and test sets.
    - **Handling Class Imbalance**: I addressed using a custom oversampling strategy for minority classes and adjusting class weights in the loss function, as traditional methods like SMOTE are not directly applicable to multi-label data.
    - **Model Experimentation**:
        - **Base Classifiers**: I chose Naive Bayes, SVM, and Logistic Regression, as they are known for their efficacy in multi-text classification.
        - **Multi-label Strategies**: I applied OneVsRestClassifier to adapt these classifiers for multi-label data. I also explored Classifier Chains (CC) and Random k-Labelsets (RAkEL) to account for label correlations, to effectively handle the large number of features and labels.

2. **spaCy NL models**
    - **Purpose**: To leverage advanced NLP capabilities (incl PoS, NER, etc) and compare against traditional ML approaches. I also selected the pre-trained NL models, so I expect this to outperform the baseline.
    - **Data Preparation**: I converted data into spaCy's format to align with its processing pipeline.
    - **Handling class imbalance**: I oversampled minority classes to adapt to the multi-label context.
    - **Model Experimentation**:
        - **Model Sizes**: Tested different model sizes (small, medium, large) to understand the impact of model complexity on performance.
        - **Pipeline Customization**: Added textcat_multilabel to the pipeline, ensuring the model is tailored for multi-label classification.
        - **Cross Validation**: Used k-Fold to avoid overfitting.

- **Evaluation metrics**
    - I chose **F1-score (micro)** for evaluating individual label performance, providing a balance between precision and recall, crucial in the context of class imbalance.
    - I used **Hamming loss** for an overall assessment, offering a view of the fraction of incorrect label predictions, which is particularly informative in multi-label scenarios.

## Model Performance
Logistic Regression with OneVsRestClassifier (0.59 F1 Score) outperformed other ML algorithms and the spacy NL models on the validation and test sets. From the spacy NL models, the medium size performed the best with (0.48 F1 score).

1. **ML Algorithms**:
| Method                 | Classifier           | F1 Score (Micro) | Hamming Loss |
|------------------------|----------------------|------------------|--------------|
| OneVsRestClassifier    | GaussianNB           | 0.28             | 0.024        |
| OneVsRestClassifier    | LinearSVC            | 0.58             | 0.012        |
| OneVsRestClassifier    | LogisticRegression   | 0.59             | 0.014        |
| ClassifierChain        | LogisticRegression   | 0.55             | 0.017        |
| Rakel                  | LogisticRegression   | 0.52             | 0.012        |
    - OneVsRestClassifier performed better than other multi-label chain methods. This makes sense since we know from our EDA we have low label correlations and most user inputs are associated with only 1 label.

2. **spacy NL Model Sizes**:
| Model Size | F1 (micro) - Validation | Hamming Loss - Validation | F1 (micro) - Test | Hamming Loss - Test |
|------------|-------------------------|---------------------------|--------------------|--------------------|
| Small      | 0.37                    | 0.014                     | 0.36               | 0.014              |
| Medium     | 0.48                    | 0.013                     | 0.48               | 0.013              |
| Large      | 0.47                    | 0.013                     | 0.44               | 0.013              |

- **Interpretations**: It was interesting to see the ML baseline outperforming spacy NL models in terms of F1 score. Perhaps spacy models are underfitting, as they might not capture the complexity of multi-label and class-imbalanced data effectively. Another reason could be the feature representation, perhaps TF-IDF is better suited for this specific dataset compared to spacy's embeddings.

- **Future Recommendations**: Explore more advanced models (e.g. LSTM, transformers, LLM, ensemble, etc), hyperparameter tuning, and additional feature engineering and preprocessing.

## Productionization
- **API Development**: I built a FastAPI application to serve our best spacy model. I chose FastAPI over Flask due to its performance, ease of use, and automatic interactive documentation. 
    - *Side note: Here I selected the best spacy model to productionize instead of the best overall model (Logistic Regression with OVR), because I believe the spacy model performance could improve further (e.g. when added with transformer) and its Hamming loss is on par with the best overall model, so I wanted to set up the productionization for that.*
- **Containerization**: I containerized the application (i.e. the API) using Docker for consistent deployment.
- **Deployment**: I deployed it as a microservice on Azure Container Instance (ACI). I selected ACI for its simplicity and ease of use over more complex solutions like Kubernetes or Helm, considering the project's scope and scale.