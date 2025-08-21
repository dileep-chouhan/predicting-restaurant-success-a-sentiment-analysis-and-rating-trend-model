import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic data for restaurant reviews and ratings
num_reviews = 200
data = {
    'Date': pd.to_datetime(pd.date_range(start='2022-01-01', periods=num_reviews)),
    'Rating': np.random.randint(1, 6, size=num_reviews), # Ratings from 1 to 5
    'Review': [' '.join(np.random.choice(['excellent', 'good', 'average', 'poor', 'terrible'], size=np.random.randint(2, 10))) for _ in range(num_reviews)]
}
df = pd.DataFrame(data)
# --- 2. Sentiment Analysis ---
df['Polarity'] = df['Review'].apply(lambda review: TextBlob(review).sentiment.polarity)
df['Subjectivity'] = df['Review'].apply(lambda review: TextBlob(review).sentiment.subjectivity)
# --- 3. Data Cleaning and Feature Engineering ---
# (In a real-world scenario, this would involve more robust cleaning)
df['Month'] = df['Date'].dt.to_period('M')
# --- 4. Analysis ---
# Group data by month and calculate average rating and sentiment
monthly_data = df.groupby('Month')[['Rating', 'Polarity', 'Subjectivity']].mean()
# --- 5. Visualization ---
# Plot average rating over time
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(monthly_data.index.to_timestamp(), monthly_data['Rating'])
plt.title('Average Monthly Rating')
plt.ylabel('Average Rating')
# Plot average sentiment polarity over time
plt.subplot(2, 1, 2)
plt.plot(monthly_data.index.to_timestamp(), monthly_data['Polarity'])
plt.title('Average Monthly Sentiment Polarity')
plt.ylabel('Average Polarity')
plt.xlabel('Month')
plt.tight_layout()
output_filename = 'rating_sentiment_trend.png'
plt.savefig(output_filename)
print(f"Plot saved to {output_filename}")
# --- 6. (Optional) Predictive Modeling ---
#  This section would involve building a predictive model (e.g., using time series analysis or regression) 
#  to forecast future restaurant performance based on the trends observed in the data.  This is omitted for brevity.