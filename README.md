# SC1015 Academic Performance Prediction

### About
Welcome to our SC1015 Project. 
The Project is about predicting the range of academic scores of students given a data set. For a more detailed walkthrough, please look through the 2 notebooks.

### Problem Definiton: 
- What range will a student's score (low, medium or high) be given its attributes? (Multi-Classification Problem)
**Sub-problems**: 
- 1) Which Machine-Learning model is the best model to predict a student's score range?
- 2) Which attribute influences a student's score most?

### Repository content:
> 1) ProjectEDAFinal.ipynb
> 2) ProjectMLFinal 
> 3) xAPI-Edu-Data.csv 
> 4) README.md

### File Description
> ProjectEDAFinal.ipynb -- main notebook containing the main source of code for the data cleaning, data analysis and data visualisation. </br>
> ProjectMLFinal -- main notebook containing the main source of code for machine learning required for sub problem 1 and 2.</br>
> xAPI-Edu-Data.csv -- dataset we have chosen and used for this project. For more information: https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data</br>
> readme.md -- file containing basic information about the project.</br>

### Contributors:
- Ren Yu - Machine Learning Sub Problem 1
- Xu Hang - Machine Learning Sub Problem 2
- Yu Teng - Exploratory Data Analysis, Data Cleaning and Visualisation

### Models Used:
- Logistic Regression
- Decision Tree
- Random Forest Classifier
- Support Vector Machine
- K Nearest Neighbors

### Conclusion:
- We are able to quite accurately predict a student's Score Range (Low, Medium or High) with a classification accuracy of 0.75-0.85.
- RandomForestClassifier is the best model used for our classification and prediction problem.
- Number of days a student is absent for has the biggest influence on a student's score.
- Optimising parameters of classifier models give better results.

### What did we learn from the project?
- New visualisation and analysis tools
- New Machine Learning models such as KNN, SVM, Logistic Regression and Random Forest Classifiers
- The use of cross-validation to compare models
- The use of optimising parameters of a model to achieve a better classification accuracy
- How to solve a multi-classifcation problem
- How to figure out which predictor is most impactful on response variable
- Collaborating on a coding project as a group
- Collaborating using GitHub

### References
- https://inria.github.io/scikit-learn-mooc/python_scripts/03_categorical_pipeline_column_transformer.html
- https://scikit-learn.org/stable/modules/classes.html
- https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data
- https://machinelearningmastery.com/k-fold-cross-validation/
- https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
- https://quick-adviser.com/what-is-c-parameter-in-logistic-regression/
- https://machinelearningmastery.com/types-of-classification-in-machine-learning/
- https://slidesgo.com/theme/data-science-consulting


