# Retail Data Analysis & Revenue Prediction Report

## 1. Introduction

In the highly competitive retail sector, understanding the drivers of sales volume and overall revenue is crucial for strategic decision-making. This project presents an end-to-end data analysis on a simulated retail dataset. The primary goal is to uncover actionable insights regarding customer purchasing behavior, the impact of marketing (advertising spend), pricing, and seasonal events (holidays). Furthermore, we built a machine learning model to predict future revenue based on these operational metrics.

## 2. Data Preparation

Real-world data is rarely perfect. To simulate this, our initial dataset contained various inconsistencies that required cleaning and preprocessing before any analysis could begin.

> [!TIP]
> Data cleaning is often the most time-consuming part of a data science project, but it is critical. "Garbage in, garbage out" applies heavily to machine learning models.

**Key Data Cleaning Steps Performed:**
1. **Handling Missing Values:** We identified missing records in the `Revenue` column. Instead of dropping these valuable rows, we imputed (filled) the missing values using the median revenue, which is robust against extreme outliers.
2. **Removing Duplicates:** We found and removed duplicate rows that could have skewed our statistical summaries and model training.
3. **Treating Outliers:** Some `Price` values were incorrectly recorded as negative numbers (e.g., `-50`). We filtered out these erroneous records, ensuring all prices were strictly positive.

After cleaning, the dataset contained 994 valid transactions across 5 product categories, with an average transaction revenue of `$10,963.49`.

## 3. Exploratory Data Analysis (EDA)

With a clean dataset, we moved on to exploring the data visually to find patterns, trends, and anomalies.

### 3.1 Revenue Distribution
First, we looked at the overall distribution of our target variable: Revenue.

![Revenue Histogram](./revenue_histogram.png)

The distribution is relatively symmetrical, centered around the $10,000 mark. This suggests stable sales performance across the dataset without massive skewness caused by a few ultra-high-value transactions.

### 3.2 Revenue by Product Category
Next, we analyzed how revenue varies across different product categories (e.g., Electronics, Clothing, Toys).

![Revenue Boxplot](./revenue_boxplot.png)

The boxplots show consistent median revenues across categories, though the spread (variance) differs slightly. This indicates that our product portfolio is well-balanced.

### 3.3 Advertising Spend vs. Revenue
Advertising is a significant cost center, so we need to understand its impact on revenue, particularly during holiday seasons.

![Ad Spend Scatter](./adspend_scatter.png)

There is a clear positive trend: higher advertising spend generally correlates with higher revenue. Additionally, the points colored representing `Holiday_Flag = 1` (holidays) tend to sit higher on the Revenue axis, confirming that holidays boost sales efficiency.

### 3.4 Correlation Analysis
To quantify the relationships between all our numerical variables, we generated a correlation heatmap.

![Correlation Heatmap](./correlation_heatmap.png)

> [!NOTE]
> Correlation values range from -1 to 1. Values closer to 1 indicate a strong positive relationship, while values near 0 indicate no linear relationship.

**Key Observations:**
- **Price and Revenue:** Highly positively correlated. Higher-priced items naturally contribute more to total revenue.
- **Sales Volume and Revenue:** Strong positive correlation.
- **Advertising Spend and Sales Volume:** Show a positive correlation, validating the effectiveness of the marketing budget.

## 4. Predictive Modeling Results

To move from descriptive analytics (what happened) to predictive analytics (what will happen), we built a **Random Forest Regressor**. This machine learning model uses multiple decision trees to learn complex, non-linear relationships between our features (`Price`, `Discount`, `Ad_Spend`, `Holiday_Flag`) and our target (`Revenue`).

We split our data into a training set (80% of data to teach the model) and a testing set (20% of data to evaluate the model on unseen data).

**Model Performance Metrics:**
- **R-squared ($R^2$): 0.93**
  - *Interpretation:* The model explains 93% of the variance in Revenue. This is an exceptionally strong score, indicating the model is highly accurate at predicting revenue based on the input features.
- **Mean Squared Error (MSE): 2,983,259.09**
  - *Interpretation:* While this number looks large, it is the squared error of revenue dollars. The high $R^2$ confirms the model's overall reliability.

## 5. Actionable Insights

Based on our exploratory analysis and predictive modeling, we can derive the following business insights:

1. **Optimize Ad Spend During Holidays:** The scatterplot clearly shows that the combination of high ad spend and holidays yields the highest revenue peaks. Marketing budgets should be heavily weighted towards these periods to maximize Return on Ad Spend (ROAS).
2. **Price is the Primary Revenue Driver:** Given the high correlation between price and revenue, the business should explore price elasticity. Small, strategic price increases on high-volume items could significantly lift overall revenue without necessarily deterring purchases.
3. **Predictable Revenue Forecasting:** The Random Forest model's 93% accuracy means the business can reliably forecast future revenue based on planned marketing budgets and pricing strategies. This allows for better inventory planning and cash flow management.

## 6. Conclusion

This project successfully demonstrated an end-to-end data science workflow on retail data. By cleaning dirty data, visualizing key relationships, and building a highly accurate Random Forest predictive model, we transformed raw transaction logs into actionable business intelligence. The resulting model and insights provide a strong foundation for data-driven decision-making in pricing and marketing strategies.
