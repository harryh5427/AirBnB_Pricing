# Prediction of Optimal Price of AirBnB Listings in New York City
The nightly price of an AirBnB listing is a critical factor for guests when selecting their accommodations. Hosts need to set the right price to ensure the success of their listings. By analyzing the prices of highly-rated listings, hosts can determine the optimal price for their listings based on factors such as location, maximum number of guests, room type, and Wi-Fi availability. In this notebook, we present regression models that predict the optimal nightly price of AirBnB listings in New York City. Using these models, we provide hosts with a basis for maximizing their revenue based on their specific listing features and conditions.

Our best performing model is the XGBoost Regressor, which predicts that 92% of unhealthy listings in NYC can increase their monthly revenue by an average of $948 if they adjust their price to the optimal level. Our analysis provides insights into various business questions, such as how hosts and guests can determine if a listing is over or underpriced, whether a listing lacks popular amenities compared to similar listings, and why a listing may receive few bookings despite being located in a popular tourist area. We also investigate the effect of each listing feature on the predicted price, enabling hosts to determine where to invest their resources to maximize their revenue.

In `Harry_Han_AirBnBPricing_FeatureEngineering.ipynb`, we perform feature engineering on an AirBnB dataset in order to create models that predict the optimal price of AirBnB listings. The [dataset](http://data.insideairbnb.com/united-states/ny/new-york-city/2022-12-04/data/listings.csv.gz) contains detailed information on AirBnB listings in New York City and was collected in December 2022 by [Inside AirBnB](http://insideairbnb.com/get-the-data/). We explore each feature of the dataset, extract essential information, and transform it into a form that can be passed to regression models.

In `Harry_Han_AirBnBPricing_Main.ipynb`, we load the dataset which was processed in `Harry_Han_AirBnBPricing_FeatureEngineering.ipynb`, and train various regression models. We apply the best performing model, XGBoost regressor, to predict optimal price of healthy and unhealthy listings, and analyze the impact of each listing feature in predicting optimal prices. We take a few examples from overpriced and underpriced unhealthy listings, and see which features make their price not ideal.
