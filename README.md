# Prediction of Optimal Price of AirBnB Listings in New York City
This project aims to predict the optimal nightly price of AirBnB listings in New York City using regression models. The project includes the following files:

- `Harry_Han_AirBnBPricing_Main.ipynb`: the main notebook
- `Harry_Han_AirBnBPricing_FeatureEngineering.ipynb`: the notebook containing the feature engineering process
- `train.py`: the script for training models
- `run_train.sh`: the script for running `train.py`
- `environment.yml`: the conda environment for the project
- `data`: the folder containing csv data files for the project
- `model`: the folder containing joblib files of the trained models
- `figures`: the folder containing the generated figure files from the notebooks

The nightly price of an AirBnB listing is a critical factor for both guests and hosts. In this project, regression models are developed based on the analysis of the prices of highly-rated listings. By considering factors such as location, maximum number of guests, room type, and Wi-Fi availability, hosts can determine the optimal price for their listings.

The XGBoost Regressor is identified as the best performing model, predicting that 92% of unhealthy listings in NYC can increase their monthly revenue by an average of $948 by adjusting their price to the optimal level. Additionally, the project provides insights into various business questions such as determining if a listing is over or underpriced, identifying popular amenities compared to similar listings, and understanding why a listing may receive few bookings despite being located in a popular tourist area. The effect of each listing feature on the predicted price is also investigated, allowing hosts to determine where to invest their resources to maximize their revenue.

In `Harry_Han_AirBnBPricing_FeatureEngineering.ipynb`, the [AirBnB dataset](http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/data/listings.csv.gz) of listings in New York City in December 2022, obtained from [Inside AirBnB](http://insideairbnb.com/get-the-data/), is explored, and feature engineering is performed to prepare the data for regression models.

In `Harry_Han_AirBnBPricing_Main.ipynb`, the preprocessed dataset is loaded and various regression models are trained. The best performing model, XGBoost Regressor, is applied to predict optimal prices of healthy and unhealthy listings, and the impact of each listing feature in predicting optimal prices is analyzed. Overpriced and underpriced unhealthy listings are also explored to understand the features contributing to their pricing.
