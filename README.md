Iris Flower Prediction:

This project aims to predict the species of iris flowers based on their sepal and petal characteristics using machine learning techniques.

Overview:

The Iris Flower Prediction project utilizes the famous Iris dataset, which contains samples of iris flowers belonging to three different species: Setosa, Versicolor, and Virginica. Each sample includes four features: sepal length, sepal width, petal length, and petal width. The goal is to train a machine learning model that accurately classifies iris flowers into their respective species based on these features.

Data Exploration:

The dataset used in this project contains no missing values (NaN or null), making it suitable for analysis and model training. Exploratory data analysis (EDA) was performed using matplotlib and seaborn libraries to visualize the distributions and relationships between different features.

Model Training:

Logistic regression was chosen as the classification algorithm due to its simplicity and effectiveness for binary and multiclass classification tasks. The scikit-learn library was utilized to train the logistic regression model on the iris dataset.

The accuracy of the trained model was found to be 100%, indicating that it can effectively differentiate between the three iris species based on their sepal and petal characteristics.

Usage:

To use the trained model for prediction, simply input the sepal length, sepal width, petal length, and petal width of an iris flower into the provided interface. The model will then predict the species of the iris flower based on these input values.

Repository Structure":

iris.csv: Contains the Iris dataset used for training and testing the model.

Prediction of iris flower: Jupyter notebooks used for data exploration, model training, and evaluation.

iris.py: the python code that contains streamlit app



Getting Started:

To get started with this project, follow these steps:

Clone this repository to your local machine.

Navigate to the project directory.

Install the required dependencies using pip install.

Run the provided Jupyter notebooks to explore the data and train the model.

Utilize the trained model for iris flower prediction using the provided interface.

Dependencies
Juypter notebook
python
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
Streamlit


Acknowledgments

Special thanks to the scikit-learn development team for providing an easy-to-use machine learning library in Python.
