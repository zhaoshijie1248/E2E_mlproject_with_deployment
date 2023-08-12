# End-to-End Machine Learning Project with Deployment #
## Tech Stack: Python, Flask, Docker, AWS ECR/EC2, CI/CD

## Objective

In the rapidly evolving landscape of technology, state-of-the-art machine learning models continue to emerge incessantly. For a data scientist, the ability to craft models that align with practical business needs is indispensable. However, the process doesn't stop there; deploying these models effectively is of equal significance. 

This project is meticulously crafted to provide hands-on experience across the complete lifecycle of a machine learning endeavor. From the initial stages of data ingestion and exploratory data analysis to the intricacies of model training, prediction, and development, the journey also encompasses crucial aspects like logging, exception handling, and thoughtful project structure design.

## Problem statement
- This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.


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
