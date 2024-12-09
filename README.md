**Developer** - Sathvika Reddy Alapty
**Collaborators** - None

## Wine Quality Prediction Using Spark and AWS
- This project demonstrates the implementation of a Machine Learning model for predicting wine quality using Apache Spark MLlib. The solution leverages Amazon Web Services (AWS) for distributed processing with EMR clusters and Docker for containerization.

### Project Overview
- Objective: Train a machine learning model to predict wine quality based on a dataset of wine characteristics.
- Technologies Used:
    - Apache Spark MLlib for machine learning.
    - AWS EMR for distributed data processing.
    - Docker for containerization.
    - S3 for storing datasets and models.

### Features
- Training Model: Train a predictive model using the provided training dataset.
Store the trained model in an S3 bucket for reuse.
- Prediction: Evaluate the model on a validation dataset to compute accuracy and F1 score.
- Containerization: Package the solution in a Docker container for easy deployment.
- AWS EMR Integration: Utilize an EMR cluster to process training and predictions in a distributed manner.

### Setup Instructions
- Prerequisites:
    - An AWS account with permissions for EMR, EC2, and S3 services.
    - Docker installed on your machine.
    - Python environment prepared with necessary dependencies.
- Set Up AWS EMR:
    - Create an S3 bucket and upload all project files: Training.py, Prediction.py, TrainingDataset.csv, and ValidationDataset.csv.
    - Create an EMR cluster with one master node and four task nodes.
    - Ensure security settings allow access to the master node from your IP address.
- Run the Training and Prediction Tasks:
    - Use Apache Spark to train the model and store it in the S3 bucket.
    - Validate the model's predictions and compute performance metrics.
- Dockerize the Application:
    - Build and test a Docker image containing the training and prediction scripts.
    - Push the Docker image to DockerHub for deployment.
### File Structure
- Training.py: The script for training the machine learning model.
- Prediction.py: The script for making predictions using the trained model.
- TrainingDataset.csv: The dataset used for training the model.
- ValidationDataset.csv: The dataset used for validating the model.
- Dockerfile: The configuration file for building the Docker image.