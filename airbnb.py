import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('airbnb_data3.csv')
X = data[['accommodates', 'bedrooms', 'bathrooms', 'number_of_reviews']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

fig, ax = plt.subplots()
bar_width = 0.35
opacity = 0.8

index = np.arange(len(y_test))

rects1 = ax.bar(index, y_test, bar_width,
                alpha=opacity, color='green',
                label='Actual Price')


rects2 = ax.bar(index + bar_width, y_pred, bar_width,
                alpha=opacity, color='red',
                label='Predicted Price')

#Actual vs Predict Price Graph
ax.set_xlabel('Listing')
ax.set_ylabel('Price')
ax.set_title('Price Prediction: Actual vs Predicted')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(index)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


data['Sentiment'] = data['Review'].apply(lambda x: TextBlob(x).sentiment.polarity)


data['Sentiment_Label'] = data['Sentiment'].apply(lambda x: 'Positive' if x > 0 else 'Negative' if x < 0 else 'Neutral')

average_sentiment = data['Sentiment'].mean()

print('Sentiment Analysis Results:')
print(data[['Review', 'Sentiment', 'Sentiment_Label']])
print('Average Sentiment Score:', average_sentiment)

# Plot the sentiment distribution as a pie chart
sentiment_counts = data['Sentiment_Label'].value_counts()
colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'blue'}

plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=[colors[label] for label in sentiment_counts.index], autopct='%1.1f%%')
plt.title('Sentiment Distribution')
plt.axis('equal')
plt.show()

airbnb_data = pd.read_csv("airbnb_data3.csv")
best_locality = airbnb_data.groupby("neighborhood").agg({"locality": "count", "price": "mean"})
best_locality = best_locality.sort_values("locality", ascending=False).head(10)

#Neighborhood Analysis
plt.figure(figsize=(12, 6))
plt.plot(best_locality.index, best_locality["price"], marker='o', linestyle='-', color='b')
plt.xlabel("Neighborhood")
plt.ylabel("Average Price")
plt.title("Neighborhood Analysis: Best Locality and Price")
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

#Host Performance Analysis
data = pd.read_csv('airbnb_data4.csv')

# Calculate the total number of listings per host
listings_per_host = data.groupby('host_id')['listing_id'].count()

# Calculate the average rating per host
avg_rating_per_host = data.groupby('host_id')['rating'].mean()

# Calculate the average price per host
avg_price_per_host = data.groupby('host_id')['price'].mean()

# Calculate the total revenue per host
total_revenue_per_host = data.groupby('host_id')['revenue'].sum()

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot listings per host
axes[0, 0].bar(listings_per_host.index, listings_per_host.values)
axes[0, 0].set_xlabel('Host ID')
axes[0, 0].set_ylabel('Number of Listings')
axes[0, 0].set_title('Listings per Host')

# Plot average rating per host
axes[0, 1].bar(avg_rating_per_host.index, avg_rating_per_host.values)
axes[0, 1].set_xlabel('Host ID')
axes[0, 1].set_ylabel('Average Rating')
axes[0, 1].set_title('Average Rating per Host')

# Plot average price per host
axes[1, 0].bar(avg_price_per_host.index, avg_price_per_host.values)
axes[1, 0].set_xlabel('Host ID')
axes[1, 0].set_ylabel('Average Price')
axes[1, 0].set_title('Average Price per Host')

# Plot total revenue per host
axes[1, 1].bar(total_revenue_per_host.index, total_revenue_per_host.values)
axes[1, 1].set_xlabel('Host ID')
axes[1, 1].set_ylabel('Total Revenue')
axes[1, 1].set_title('Total Revenue per Host')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()


