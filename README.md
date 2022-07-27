# Malaysia Covid 19 Case Prediction
## Project Description 
This is a time series prediction model to predict the trend of the covid 19 cases in Malaysia. The project dataset is split into training data set and testing dataset in the uploaded folder named 'datasets'. The datasets has some missing value and has been treated with polynomial interpolation. 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
 ![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
 ![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
 ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
 ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
 ![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)

# How to Install and Run the Project 
If you wish to run this model on your pc, you may download the train and test dataset from the "datasets" folder together with Covid19_train.py and Covid_module.py file from the depository section. The generated models from this dataset has been uploaded for you in 2 separated folder named "model" in the depository section. 

Software required: Spyder, Python(preferably the latest version) 

Additional modules needed: Tensorflow, Sklearn, matplotlib

# Architecture of the model 
![alt text](https://github.com/CHuiV123/Malaysia-Covid-19-Case-Prediction/blob/d031a0e5be3e22a639056adfd0d5f06d2af8d0d5/static/model.png)

# Outcome 
## Tensorboard Display 
![alt text](https://github.com/CHuiV123/Malaysia-Covid-19-Case-Prediction/blob/d031a0e5be3e22a639056adfd0d5f06d2af8d0d5/static/Tensorboard.png)

## Training MSE vs Validation MSE
![alt text](https://github.com/CHuiV123/Malaysia-Covid-19-Case-Prediction/blob/d031a0e5be3e22a639056adfd0d5f06d2af8d0d5/static/Figure%202022-07-27%20185153%20(0).png)

## Actual Cases VS Predicted Cases  
![alt text](https://github.com/CHuiV123/Malaysia-Covid-19-Case-Prediction/blob/d031a0e5be3e22a639056adfd0d5f06d2af8d0d5/static/Figure%202022-07-27%20185153%20(1).png)

## MAE, MSE and MAPE  
![alt text](https://github.com/CHuiV123/Malaysia-Covid-19-Case-Prediction/blob/089106031c23557c049245138f840115b22dcc56/static/MAE%20MSE%20MAPE.png)

## Conclusion 
This prediction model is able to show a good catch of the trend of Covid-19 cases in Malaysia. It generally shows a downward trend indicating positive cases will drop for days to come in Malaysia. 



## Credits
This datasets is provided by [Ministry of Health Malaysia] [https://github.com/MoH-Malaysia/covid19-public] 
