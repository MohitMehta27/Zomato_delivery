# Delivery Time Prediction
=======================================================================

* I have developed a state-of-the-art machine learning model that is capable of accurately predicting the delivery time of the delivery person. Additionally, I have implemented modular coding techniques to streamline the pipelines, allowing the system to be executed using a single python file. Furthermore, The code is able to generate artifacts and logs, providing the  valuable insights into its performance.





1) install the requriments

```bash
pip install -r requirements.txt
```

2) run(By running this file artifacts will automatically generated)

```bash
python src/components/pipeline/training_pipeline.py
```

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── artifacts          <- For Saving model and processor pipeline pickle files
    │
    ├── notebooks          <- Jupyter notebooks
    │                     
    │                        
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   │
    │   ├── data_ingestion <- Scripts to turn raw data into features for modeling and data transformation
    |   |   ├── data_ingestion.py
    │   │   
    │   │
    │   ├── data_transformation <- Scripts to turn raw data into features for modeling and data transformation
    |   |   ├── data_ingestion.py

    │   ├── model_trainer         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    |   ├── pipeline       <- Pipelines to train train and predict
    │   │   │
    │   │   ├── prediction_pipeline.py
    │   │   └── training_pipeline.py
    |   |
    │   |
    |   ├── exception.py   <- Script handle sys exceptions
    |   |
    |   ├── logger.py      <- Script handle logging data to logs
    |   |                  
    |   └── utils.py


