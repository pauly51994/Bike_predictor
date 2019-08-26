# Bike Rental Count Model

## Goal
Predict bike rental count on any given day (bike count is the dependent variable)

#### The Data
Pulled data from UCI's Machine Learning Repository. This dataset is on the Capital Bikeshare System located in the Washington, D.C. area, from the year 2011 to 2012. The main components of the dataset used were the humidity, temperature, weather, season and working day. The UCI repository gathered this data from Capital Bikeshare's website and the weather data from i-weather.com. 

- temperature was normalized in Celsius
- humidity was normalized out of 100
- season was categorized as summer, fall, winter, spring
- weather was categorized as clear, misty, light storm and heavy storm
- working day was 0 or 1 (whether it was working day or not)

#### Questions
- How many bikes will be rented on a given day?
- Which features would influence bike count the most?
- Can we accurately predict bike count based on the selected features?

#### Why bike count?
- Determine distribution of bikes through the system based on predicted demand
- Profit potential based on bike count under particular conditions
- Possibly increase advertisement during high bike count days

#### Methods and Libraries used
- ScikitLearn for linear regression
- Pandas for dataframes
- Statsmodels for OLS/model creation
- Matplotlib for data visualizations

## Process
1. Set up dataframe
2. EDA (Exploratory Data Analysis)
3. Model creation and testing

#### 1. Set up Dataframe
The UCI dataset provided a csv file with daily information on weather status, date, and bikes rented. With weather status and season, dummy vairables were created because of their categorical nature in order to make the data usable for a model. Season was determined to be inaccurate, so seasons were assigned based on the month to be more accurate to reality. There was one outlier in the dataset where only 22 bikes were rented on that day (this was due to Hurricane Sandy). The outlier was removed. The final dataframe was exported to a JSON file for later use.

Notebook to see dataframe set up process: (df etup link)

#### 2. EDA (Exploratory Data Analysis)
(input plot of original data w/ regression line)

This data is inherently a time-series dataset with seasonality, which displayed an upward trend with time. To account for this upward trend, we normalized the data by using ScikitLearn to plot a simple linear regression line and worked with the residuals between the line and the actual bike count (residuals = actual bike count - regression line predictions). These residuals became our new dependent variable.

(input resids plot)

Notebook to see EDA process: (link)

#### 3. Model creation and testing
Looking through our data, with so many categorical variables, we decided to first create interaction variables between all features and see which ones were statistically significant at a 95% confidence interval. To do so, we used Statsmodels's OLS method to create our first model with all these variables and checked each one's p-value. We cut out every variable that had a p-value of 0.9 or higher, and then ran the model again using the remaining variables. We repeated this process until there were no variables with p-values above the 0.9 threshold. From there, we lowered the cut-off threshold to 0.8 and reran the model. We did this process multiple times over, until the threshold was 0.05, and all the remaining variables were statistically significant.

This entire process cut down our dependent and interaction variable count from over 240 down to 41 variables.

Notebook to see model creation process: (link model notebook)


## Final Model
Our final model predicts the residuals of the original simple linear regression line. We added the predicted residuals of our final model to the linear regression's predictions of bike count to see how well our model predicts bike count. Below is a side-by-side comparison of our model predictions and actual bike count.

(input model visuals)
