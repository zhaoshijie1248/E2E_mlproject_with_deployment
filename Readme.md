# Deploying ML models as web service to cloud #

Serving a machine learning model as a web service

Tech stack: Python, Flask, Docker, AWS ECR/EC2, CI/CD


## Objective
Be acquainted with development workflow, and related tech stack, and get the first taste of building an ML production system.

## Training a machine learning model on a local system
1. [EDA STUDENT PERFORMANCE.ipynb](https://github.com/zhaoshijie1248/E2E_mlproject_with_deployment/blob/main/notebook/1%20.%20EDA%20STUDENT%20PERFORMANCE%20.ipynb): exploratory data analysis and visualization
2. [data_ingestion.py](https://github.com/zhaoshijie1248/E2E_mlproject_with_deployment/blob/main/src/components/data_ingestion.py): read data and split them into train set and test set
3. [data_transformation.py](https://github.com/zhaoshijie1248/E2E_mlproject_with_deployment/blob/main/src/components/data_transformation.py): standard scale on numerical features and do one-hot encoding on categorial features
4. [model_trainer.py](https://github.com/zhaoshijie1248/E2E_mlproject_with_deployment/blob/main/src/components/model_trainer.py): apply classification algorithms, including Random Forest, Decision Tree, Gradient Boosting, Linear Regression, XGBRegressor, CatBoosting Regressor, AdaBoost Regressor, to train data and select the model with the best performance to save as ['model.pkl'](https://github.com/zhaoshijie1248/E2E_mlproject_with_deployment/blob/main/artifacts/model.pkl) in 
    ```
        curl -X POST \
        0.0.0.0:80/predict \
        -H 'Content-Type: application/json' \
        -d '[5.9,3.0,5.1,1.8]'
    ```

## Data Collection
- Dataset Source - https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977
- The data consists of 8 column and 1000 rows.

## Data Exploration

- Check Missing values
- Check Duplicates
- Check data type
- Check the number of unique values of each column
- Check statistics of data set
- Check various categories present in the different categorical column

## Data Visualization
  
- **Model Development**: It will delve into the process of creating and fine-tuning machine learning models using Python. Various libraries and frameworks will be explored to achieve this.

- **Deployment**: This project emphasizes the deployment phase, where the focus will be on using Flask to create a web service for the trained model. Docker will be employed to containerize the application, ensuring consistent deployment across environments.

- **Cloud Integration**: The project will guide you through utilizing AWS ECR/EC2 for hosting the Dockerized application, allowing for efficient scaling and management.

- **Continuous Integration and Continuous Deployment (CI/CD)**: You will learn how to set up CI/CD pipelines to automate the testing, building, and deployment processes. This ensures that changes to the codebase can be seamlessly integrated and deployed, reducing manual intervention.

## Getting Started

Follow these steps to get started with the project:

1. Clone this repository to your local machine.
2. Install the required dependencies listed in the `requirements.txt` file.
3. Prepare your dataset or use a sample dataset to experiment with the model development process.
4. Refer to the documentation provided for each phase of the project to understand the workflow.
