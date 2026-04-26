## B1. Problem Formulation

# (a)
This problem can be formulated as a supervised machine learning regression task.

The target variable is items_sold, which represents the number of items sold at a store during a given period.

The candidate input features include:
- Store characteristics: store_size, location_type
- Promotion details: promotion_type
- Temporal features: transaction_date (or derived features such as month, day of week, weekend, and festival indicators)
- External factors: competition_density

This is a regression problem because the objective is to predict a continuous numerical value (number of items sold). Regression is appropriate since we are estimating sales volume rather than assigning categories.

# (b)
We use items_sold (sales volume) instead of total sales revenue as sales revenue can be affected by price, discounts and other promotional activity. For example, a promotion which drives up the number of items sold might actually decrease sales revenue due to lower selling price.

In contrast, items_sold directly reflects customer demand and response to promotions, making it a more stable and meaningful target variable.

This illustrates the broader principle that the target variable should closely align with the business objective and should not be influenced by external or confounding factors. In real-world ML projects, selecting a clear, consistent, and interpretable target variable is essential for building reliable models.

# (c)
Instead of using one global model, a better approach might be to use a segmented or hierarchical modeling strategy.

For example, the separate models can be built for different store groups based on location type such as urban, semi-urban, rural or based on store size. Alternatively, location_type can be included as a key feature in the model so that it can learn the particular characteristics of different store categories.

This approach is justified because customer behavior, purchasing power, and response to promotions vary significantly across locations. A single global model may fail to capture these differences, leading to poor predictions.

By accounting for store-level variations, the model can provide more accurate and context-specific insights, resulting in getting better decision-making for promotion strategies.


## B2. Data and EDA Strategy

# (a)
The four tables (transactions, store attributes, promotion details, and calendar) would be joined using common keys:

- transactions joined with store attributes using `store_id`
- transactions joined with promotion details using `promotion_type` or promotion ID
- transactions joined with calendar using `transaction_date`

The final dataset would have a grain of:
one row per store per day per promotion

Before modelling, aggregation would be required:
- Aggregate transactions to calculate total items_sold per store per day
- Ensure one unique record per store-date combination
- Merge store-level attributes (store_size, location_type)
- Merge calendar features (weekend, festival)
- Include promotion-related features

This ensures the dataset is structured at a consistent level suitable for modelling.

# (b)
Before building the model, the below EDA steps might be performed:

1. Distribution of target variable (items_sold)
   - Use histogram or boxplot
   - Check for skewness or outliers
   - If highly skewed → consider log transformation

2. Sales by promotion type
   - Use bar chart
   - Compare average items_sold across promotions
   - Helps identify which promotions are more effective

3. Sales by store characteristics
   - Group by store_size and location_type
   - Analyze average sales patterns
   - Helps decide whether segmentation or interaction features are needed

4. Time-based trends
   - Line plot of sales over time
   - Identify seasonality, weekend effects, or festival effects
   - Helps in feature engineering (e.g., month, day_of_week)

5. Correlation analysis
   - Heatmap of numerical features
   - Identify relationships and multicollinearity
   - Helps in feature selection and model choice

These analysis help understand patterns in the data and guide feature engineering, model selection, and preprocessing decisions.

# (c)
The imbalance (80% transactions without promotion) can bias the model to learn patterns dominated by non-promotional data, reducing its ability to accurately capture the impact of promotions.

To address this:
- Include promotion_type as an important feature
- Use sampling techniques (oversample promotional data or undersample non-promotional data)
- Apply model weighting to give more importance to promotional cases

This ensures the model learns meaningful patterns for both promotional and non-promotional scenarios.


## B3. Model Evaluation and Deployment

# (a)
The data is monthly store-level data across three years, thus a time-based approach to splitting between training set and test set would be more appropriate than random.

For example, the first 80% of months can be used for training, and the most recent 20% of months can be used for testing. This ensures that the model is trained on past data and evaluated on future data.

We chose not to use a random split, as splitting the data randomly can lead to data leakage, where future information is used to predict past outcomes. This would make the model performance look better than it actually is in a real business setting.

Evaluation metrics:
- MAE: Shows the average absolute error in items sold. It is easy for business teams to interpret.
- RMSE:  Errors are penalized more for larger differences, which is good for when large sales prediction errors are costly.
- R² Score: Shows how much variation in items_sold is explained by the model.

# (b)
To understand why the model recommends different promotions for Store 12 in different months, I would analyze the most important features influencing the prediction.

For December, the model may give higher importance to features such as festival season, month, customer demand, or weekend effects, which could make Loyalty Points Bonus more effective. For March, different conditions such as lower festival impact, different seasonal demand, or promotion response patterns may make Flat Discount more suitable.

I would compare the feature importance values and prediction drivers for both months and explain to the marketing team that the model is not recommending promotions only based on the store, but also based on month-specific and context-specific factors.

This helps communicate that different recommendations for the same store are reasonable because customer behavior can change across months.

# (c)
For deployment, the trained model should be saved using tools such as joblib or pickle so that it can be reused without retraining every month.

At the start of each month, new data for all 50 stores should be prepared in the same format as the training data. This includes store attributes, promotion options, calendar features, competition density, and any engineered date features. The saved preprocessing pipeline should be applied to the new data before generating predictions.

The model can then predict expected items_sold for different promotion options for each store, and the promotion with the highest predicted sales can be recommended.

Monitoring should include:
- Comparing predicted items_sold with actual sales after the month ends
- Tracking MAE and RMSE over time
- Checking whether data patterns have changed
- Monitoring feature distributions for data drift

If prediction errors increase significantly or customer behavior changes, the model should be retrained using updated data.
