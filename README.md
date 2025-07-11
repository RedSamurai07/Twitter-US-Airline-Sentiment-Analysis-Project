# Twitter-US-Airline-Sentiment-Analysis-Project

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Data Analysis](#data-analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
- This project focuses on analyzing customer feedback from Twitter to understand sentiment towards major airlines. The dataset provides pre-classified sentiments (positive, neutral, negative) and, for negative feedback, specific reasons for dissatisfaction. This analysis aims to provide actionable insights for airlines to enhance customer service and operational efficiency.

### Executive Summary
- This analysis of Twitter interactions provides critical insights into customer sentiment, common pain points, and behavioral patterns, equipping various airline departments with actionable intelligence to enhance service, optimize operations, and refine strategic initiatives.

**Overall Findings:**
- The data reveals a prevalent negative sentiment towards airlines, primarily driven by customer service and operational failures. While negative feedback is widespread across locations and timezones, most individual complaints do not gain significant traction through retweets. Significant data gaps exist in location and timezone information, limiting granular analysis.

**1. Customer Service Department:**
- This analysis provides the customer service team with a clear understanding of the most frequent complaints, including core "Customer Service Issues" and "Miscellaneous" problems (e.g., online cancellation difficulties, food service, long wait times). It identifies peak hours and days (mornings, nights, Sundays) for customer dissatisfaction, enabling optimized staffing and proactive engagement. Regional hotspots for negative sentiment (e.g., Washington D.C., NYC, Chicago) and specific issues within those areas (e.g., flight attendant complaints in Boston, cancelled flights in Austin, TX) allow for targeted training and resource allocation to improve response times and resolution quality.

**2. Operations Department:**
- For the operations team, the analysis highlights critical operational failures such as "Late Flights" (especially for American, United) and "Cancelled Flights" (Austin, TX). "Booking problems" are specifically identified with United Airways. Understanding these recurring issues, their frequency, and associated locations allows operations to prioritize improvements in scheduling, maintenance, ground services, and online system reliability to reduce disruptions and enhance passenger experience.

**3. Marketing & Communications Team:**
- The marketing and communications team gains valuable insights into brand perception. The analysis reveals that United, US Airways, and American Airlines face the most negative sentiment, while Virgin America enjoys the highest positive tweet proportion. This information is crucial for developing targeted communication strategies, managing public relations during peak complaint periods, and crafting messages that address common pain points. It also informs the strategic timing of campaigns, avoiding peak negative tweeting times and leveraging quieter periods (Wednesdays, Thursdays) for promotional efforts.

**4. Senior Management & Strategy:**
- This comprehensive overview allows senior management to grasp the overall state of customer satisfaction and identify systemic issues. The pervasive negative sentiment underscores the need for a holistic, customer-centric approach to strategy. Understanding the high confidence in negative classifications, the dominance of customer service and operational issues, and the impact of data gaps (no location/timezone) can guide investment in technology, staff training, and data infrastructure. It also provides a benchmark for monitoring the effectiveness of strategic initiatives aimed at improving customer experience and brand reputation.

### Goal
- The objective of this analysis is to leverage tweet data to understand public perception of airlines, identify key drivers of negative sentiment, and help airlines make data-driven decisions to improve customer satisfaction and service quality. This could involve building predictive models for sentiment or root cause analysis for the below mentioned key points.

  - Customer Loyalty & Retention
  - Demographic & Geographic Analysis
  - Program Effectiveness & Customer Behavior
  
### Data structure and initial checks
[Dataset](https://docs.google.com/spreadsheets/d/1EmudVOp_6vISH8C27vD4agJJEUKdFzSddlHsfkjXDeg/edit?gid=639920194#gid=639920194)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| tweet_id | A unique numerical identifier for each tweet. | int |
| airline_sentiment | The categorized sentiment of the tweet towards the airline. | object |
| airline_sentiment_confidence | A numerical value (between 0 and 1) indicating the confidence level of the assigned airline_sentiment. | float |
| negativereason | If the airline_sentiment is 'negative', this column specifies the reason for the negative feedback.| object  |
| negativereason_confidence | A numerical value (between 0 and 1) indicating the confidence level of the assigned negativereason. | float |       
| airline | The name of the airline mentioned in the tweet | object |
| airline_sentiment_gold | Gold-standard sentiment, likely used for a small subset of hand-labeled data for validation purposes. This column contains many missing values. | object |
| name | The Twitter handle (username) of the individual who posted the tweet. | object |
| negativereason_gold | The full content of the tweet. This is the primary data for sentiment analysis. | object |
| retweet_count | The number of times the tweet has been retweeted. | int|  
| text  | The full content of the tweet. This is the primary data for sentiment analysis. | object |
|  tweet_coord | Geographic coordinates of the tweet, if available. | object |
| tweet_created | The date and time when the tweet was posted. | object |
| tweet_location | The user-provided location from which the tweet was sent. | object |
| user_timezone | The timezone setting of the user who posted the tweet. | object |

### Tools
- Excel : Google Sheets - Check for data types, Table formatting
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation and Analysis(Numpy, Pandas),Visualization (Matplotlib, Seaborn), Feature Engineering, Hypothesis Testing
  
### Data Analysis
1). Python
- Importing Libraries
``` python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
```
- Loading the dataset
``` python
  df = pd.read_csv('Tweets.csv')
  df.head()
```
![image](https://github.com/user-attachments/assets/6a4004cd-acca-429c-9dc9-6c9ceaf79326)
- Dimension and Shape of the dataset
``` python
  df.ndim
```
![image](https://github.com/user-attachments/assets/2bdff19b-9bfd-4d10-b828-faf1960f1a3d)
``` python
df.shape
```
![image](https://github.com/user-attachments/assets/b71a28d0-73a4-4fb7-8466-fdd7ee84739c)
- Information of the Dataset
``` python
df.info()
```
![image](https://github.com/user-attachments/assets/2d766618-8142-4bc8-bef2-a32e035938f2)
- Data Cleaning and Pre-processing
``` python
   df.isna().sum()/len(df)*100
```
![image](https://github.com/user-attachments/assets/dfac95eb-0864-40c9-a282-a688ac59f056)
``` python
df.drop_duplicates(inplace = True)
```
``` python
  df.drop('airline_sentiment_gold',axis = 1,inplace = True)
  df.drop('negativereason_gold',axis = 1,inplace = True)
  df.drop('tweet_coord',axis = 1, inplace = True)
```
``` python
  df['negativereason'] = df['negativereason'].fillna('Others')
  df['negativereason_confidence'] = df['negativereason_confidence'].fillna(df['negativereason_confidence'].mean())
``` python
  df['tweet_location'].fillna('No location',inplace = True)
  df['user_timezone'].fillna('No Timezone',inplace = True)
```
``` python
  df[['Sub_airline', 'Comments']] = df['text'].str.split(' ', n=1, expand=True)
  correct_air = df['Sub_airline'] == df['airline']
  correct_air.value_counts()
```
![image](https://github.com/user-attachments/assets/8313ffca-32ac-488f-95d9-db11f74e7ad6)
``` python
  df.drop('text',axis = 1, inplace = True)
  df.drop('Sub_airline',axis = 1, inplace = True)
  df.drop('Comments',axis = 1, inplace = True)
```
``` python
  df.isna().sum()/len(df)*100
```
- Descriptive Statistics
``` python
  df.describe()
```
![image](https://github.com/user-attachments/assets/f075c519-857e-46c7-b359-378e093861b5)
``` python
  df.head()
```
![image](https://github.com/user-attachments/assets/6812260b-e0da-4b88-9fa7-d197ea3b1593)

1. **Customer Loyalty & Retention**
``` python
  pd.crosstab(df['airline'],df['negativereason'])
```
![image](https://github.com/user-attachments/assets/55fa77eb-cc45-424b-9dc6-7912cf562188)
``` python
  fig = plt.figure(figsize=(15,8))
  sns.countplot(x = 'airline',hue = 'negativereason',data = df)
  plt.title('Count of Negative Reasons by Airline')
  plt.xlabel('Airline')
  plt.ylabel('Count')
  plt.show()
```
![image](https://github.com/user-attachments/assets/11d78731-90b8-4751-90cb-c8ce030126cf)
``` python
  df['negativereason'].value_counts().plot(kind = 'bar',color = sns.color_palette('magma'))
  plt.title('Reasons for Negative Tweet')
  plt.xlabel('Negative reason')
  plt.ylabel('Count')
  plt.show()
  print(df['negativereason'].value_counts().reset_index())
```
![image](https://github.com/user-attachments/assets/f99cb6fb-cbfb-4390-a794-70cfd58b3adb)
![image](https://github.com/user-attachments/assets/109391b8-2e3d-4c33-9ce6-d6a81ce645a8)
``` python
  airline_sentiment_counts = df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)
  airline_sentiment_proportions = airline_sentiment_counts.apply(lambda x: x / x.sum()*100, axis=1)

  # Highest proportion of positive sentiment tweets
  most_positive_airline = airline_sentiment_proportions['positive'].idxmax()
  print(f"Airline with highest proportion of positive tweets: {most_positive_airline}")

  # Highest proportion of negative sentiment tweets
  most_negative_airline = airline_sentiment_proportions['negative'].idxmax()
  print(f"Airline with highest proportion of negative tweets: {most_negative_airline}")

  # Optionally, display the full proportion table
  print("\nProportion of sentiments per airline:")
  print(airline_sentiment_proportions.reset_index())

  figure=plt.figure(figsize=(15,8))
  airline_sentiment_proportions.plot(kind = 'bar',color = sns.color_palette('tab10'))
  plt.title('Airline Sentiment Proportion')
  plt.xlabel('Airlines')
  plt.xticks(rotation = 90)
  plt.ylabel('Percentage')
  plt.legend()
  plt.show()
```
![image](https://github.com/user-attachments/assets/b5321ef1-f574-451a-b772-d6c93f3d2306)
![image](https://github.com/user-attachments/assets/9215ebc5-6014-4345-9825-a38f3ac7518b)
``` python
  negative_tweets_df = df[df['airline_sentiment'] == 'negative']

  # Plotting the distribution of airline_sentiment_confidence for negative tweets
  plt.figure(figsize=(10, 6))
  sns.histplot(negative_tweets_df['airline_sentiment_confidence'], bins=20, kde=True)
  plt.title('Distribution of Airline Sentiment Confidence for Negative Tweets')
  plt.xlabel('Airline Sentiment Confidence')
  plt.ylabel('Count')
  plt.show()

  # Calculate the correlation between airline_sentiment_confidence and whether a tweet is negative
  df['is_negative'] = (df['airline_sentiment'] == 'negative').astype(int)

  # Calculating the correlation coefficient
  correlation = df['airline_sentiment_confidence'].corr(df['is_negative'])
  print(f"\nCorrelation between airline_sentiment_confidence and likelihood of a tweet being negative: {correlation}")

  # Interpreting the correlation coefficient:
  if correlation > 0.1:
     print("There is a weak positive correlation, suggesting higher confidence might be slightly associated with negative tweets.")
  elif correlation < -0.1:
     print("There is a weak negative correlation, suggesting higher confidence might be slightly associated with non-negative tweets.")
  else:
     print("There is a very weak or no linear correlation.")

  # Also, we can look at the mean confidence for different sentiment categories
  mean_confidence_by_sentiment = df.groupby('airline_sentiment')['airline_sentiment_confidence'].mean()
  print("\nMean Airline Sentiment Confidence by Sentiment")
  print('\t')
  mean_confidence_by_sentiment.reset_index()
```
![image](https://github.com/user-attachments/assets/f37c1c0d-aad6-45e7-b76a-6dae2383ab3c)
![image](https://github.com/user-attachments/assets/c3ad6ae4-db2f-4d34-bcc5-05185cb4f018)

2. **Demographic & Geographic Analysis**
``` python
  location_sentiment = df.groupby(['tweet_location', 'airline_sentiment']).size().unstack(fill_value=0)
  timezone_sentiment = df.groupby(['user_timezone', 'airline_sentiment']).size().unstack(fill_value=0)

  min_tweets = 50
  location_sentiment_filtered = location_sentiment[(location_sentiment['negative'] + location_sentiment['neutral'] + location_sentiment['positive']) >= min_tweets]
  timezone_sentiment_filtered = timezone_sentiment[(timezone_sentiment['negative'] + timezone_sentiment['neutral'] + timezone_sentiment['positive']) >= min_tweets]

  print("\nSentiment distribution by Tweet Location (filtered for locations with >= 50 tweets):")
  print(location_sentiment_filtered)

  print("\nSentiment distribution by User Timezone (filtered for timezones with >= 50 tweets):")
  print(timezone_sentiment_filtered)
```
![image](https://github.com/user-attachments/assets/a2acf2bb-b435-4f39-98be-3bee9078c721)
![image](https://github.com/user-attachments/assets/59289b22-17bb-4ffb-b56b-39af5a9e92a2)
``` python
  location_sentiment_proportions = location_sentiment_filtered.apply(lambda x: x / x.sum()*100, axis=1)
  timezone_sentiment_proportions = timezone_sentiment_filtered.apply(lambda x: x / x.sum()*100, axis=1)

  print("\nProportion of Sentiment by Tweet Location (filtered):")
  print(location_sentiment_proportions.sort_values(by='negative', ascending=False).head(10))
  print(location_sentiment_proportions.sort_values(by='positive', ascending=False).head(10))
  print('\n')
  print("\nProportion of Sentiment by User Timezone (filtered):")
  print(timezone_sentiment_proportions.sort_values(by='negative', ascending=False).head(10))
  print(timezone_sentiment_proportions.sort_values(by='positive', ascending=False).head(10))
```
![image](https://github.com/user-attachments/assets/9a104e92-6b9a-48ce-b34a-f5267865bfa2)
![image](https://github.com/user-attachments/assets/44bdf53d-eff5-4b37-96af-67b78ae5efe7)
``` python
  location_negativereason = df.groupby(['tweet_location', 'negativereason']).size().unstack(fill_value=0)
  timezone_negativereason = df.groupby(['user_timezone', 'negativereason']).size().unstack(fill_value=0)

  min_negative_tweets_for_analysis = 50
  location_negativereason_filtered = location_negativereason[location_negativereason.sum(axis=1) >= min_negative_tweets_for_analysis]
  timezone_negativereason_filtered = timezone_negativereason[timezone_negativereason.sum(axis=1) >= min_negative_tweets_for_analysis]

  print("\nDistribution of Negative Reasons by Tweet Location (filtered for locations with >= 50 negative tweets):")
  print(location_negativereason_filtered.apply(lambda x: x / x.sum(), axis=1).head())

  print("\nDistribution of Negative Reasons by User Timezone (filtered for timezones with >= 50 negative tweets):")
  print(timezone_negativereason_filtered.apply(lambda x: x / x.sum(), axis=1).head())
```
![image](https://github.com/user-attachments/assets/64543ddb-5328-4d44-a674-a06bc30451cb)
![image](https://github.com/user-attachments/assets/ca065ad8-2b54-4bc8-a680-3ff5d9497012)
``` python
  # To see if they vary significantly, we can look at the top reasons in different locations/timezones
  print("\nTop negative reasons by location (showing top 5 locations by total negative tweets):")
  for location in location_negativereason_filtered.sum(axis=1).sort_values(ascending=False).head(5).index:
  print(f"\nLocation: {location}")
  print(location_negativereason_filtered.loc[location].sort_values(ascending=False).head(5))

  print("\nTop negative reasons by timezone (showing top 5 timezones by total negative tweets):")
  for timezone in timezone_negativereason_filtered.sum(axis=1).sort_values(ascending=False).head(5).index:
     print(f"\nTimezone: {timezone}")
     print(timezone_negativereason_filtered.loc[timezone].sort_values(ascending=False).head(5))
```
![image](https://github.com/user-attachments/assets/67296a7c-a546-44e2-883e-6b99cccb0c76)
![image](https://github.com/user-attachments/assets/054a41e9-a4fe-49e1-a34a-bbaabc7af382)
![image](https://github.com/user-attachments/assets/43f63b04-3b00-45bd-b083-02ca371afb45)
``` python
  negative_reason_to_compare = 'Customer Service Issue'
    if negative_reason_to_compare in location_negativereason_filtered.columns:
    plt.figure(figsize=(15, 8))
    location_negativereason_filtered.apply(lambda x: x / x.sum()*100, axis=1)[negative_reason_to_compare].sort_values(ascending=False).head(10).plot(kind='bar',color = 'orange')
    plt.title(f'Proportion of "{negative_reason_to_compare}" by Location')
    plt.xlabel('Tweet Location')
    plt.ylabel('Proportion')
    plt.xticks(rotation=90)
    plt.show()

  if negative_reason_to_compare in timezone_negativereason_filtered.columns:
     plt.figure(figsize=(15, 8))
     timezone_negativereason_filtered.apply(lambda x: x / x.sum()*100, axis=1)[negative_reason_to_compare].sort_values(ascending=False).head(10).plot(kind='bar',color = 'darkblue')
     plt.title(f'Proportion of "{negative_reason_to_compare}" by Timezone')
     plt.xlabel('User Timezone')
     plt.ylabel('Proportion')
     plt.xticks(rotation=90)
     plt.show()
```
![image](https://github.com/user-attachments/assets/a47d13bb-25f5-48bb-b074-ec1b99daeb60)
![image](https://github.com/user-attachments/assets/3ebd204e-ccbf-4c6f-9d78-33022e3fc0ec)
``` python
  airline_location_sentiment = df.groupby(['airline', 'tweet_location', 'airline_sentiment']).size().unstack(fill_value=0)
  min_tweets_location_airline = 20
  airline_location_sentiment_filtered = airline_location_sentiment[airline_location_sentiment.sum(axis=1) >= min_tweets_location_airline]
  airline_location_sentiment_proportions = airline_location_sentiment_filtered.apply(lambda x: x / x.sum() * 100, axis=1)

  # To Analyze sentiment for each airline in specific locations
  for airline_to_analyze in airline_location_sentiment_proportions.index.get_level_values('airline').unique():
      print(f"\nAnalyzing sentiment performance for {airline_to_analyze} in specific locations:")
      airline_data = airline_location_sentiment_proportions.loc[airline_to_analyze]

      # Locations with highest positive sentiment proportion for the airline
      top_positive_locations = airline_data.sort_values(by='positive', ascending=False).head(10)
      print(f"\nTop 10 locations for {airline_to_analyze} with highest positive sentiment proportion:")
      print(top_positive_locations[['positive', 'negative', 'neutral']])

      # Locations with highest negative sentiment proportion for the airline
      top_negative_locations = airline_data.sort_values(by='negative', ascending=False).head(10)
      print(f"\nTop 10 locations for {airline_to_analyze} with highest negative sentiment proportion:")
      print(top_negative_locations[['positive', 'negative', 'neutral']])

      if not top_positive_locations.empty:
          plt.figure(figsize=(10, 8))
          top_positive_locations[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('Spectral'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Locations (by Positive Sentiment)')
          plt.xlabel('Tweet Location')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()

      if not top_negative_locations.empty:
          plt.figure(figsize=(10, 8))
          top_negative_locations[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('flare'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Locations (by Negative Sentiment)')
          plt.xlabel('Tweet Location')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()
```
![image](https://github.com/user-attachments/assets/99779867-6335-4c94-ac6f-c4fc542e2fde)
![image](https://github.com/user-attachments/assets/b2b86b0f-3d98-4692-8cad-d199ff4fe870)
![image](https://github.com/user-attachments/assets/984a8d96-778a-4427-8a46-c319ecbb5141)


![image](https://github.com/user-attachments/assets/115f04cd-48ae-4950-9d66-c498013264fd)
![image](https://github.com/user-attachments/assets/0d918237-6e88-4fc8-bb26-f9e9421e18f8)
![image](https://github.com/user-attachments/assets/48fc4f24-4448-41db-b19b-9d4f1e8b3f3f)


![image](https://github.com/user-attachments/assets/9d2deb6c-28f9-4e85-b7e3-9894700d1a0c)
![image](https://github.com/user-attachments/assets/d88717b3-0da1-4875-83c7-1d008bb137ca)
![image](https://github.com/user-attachments/assets/6f8b2b8f-4091-47e9-b57b-2a5e7d986ed9)


![image](https://github.com/user-attachments/assets/c8c238b2-34c7-40b6-89df-39d055ede0b0)
![image](https://github.com/user-attachments/assets/e8a5ebd7-9f4c-4a8f-b5be-a573797bb347)
![image](https://github.com/user-attachments/assets/b0a82a38-9ba8-4f0f-9388-f78204e2ae09)


![image](https://github.com/user-attachments/assets/3d77fbdd-7f9c-4827-a1c5-c52d2e0777d5)
![image](https://github.com/user-attachments/assets/42cb5536-efe1-4035-be8c-0d1a8cdb82ad)
![image](https://github.com/user-attachments/assets/127f2912-1c0f-4770-a2ce-65b940aab498)


![image](https://github.com/user-attachments/assets/8251805d-d637-4382-a1db-fb7e14d9bd65)
![image](https://github.com/user-attachments/assets/7d4bcdcf-5414-43a2-a9d2-e783c18f3170)
![image](https://github.com/user-attachments/assets/fcb3d697-c872-478c-8f76-be9980d69efa)
``` python
  # To Analyze sentiment for each airline in specific timezones
  airline_timezone_sentiment = df.groupby(['airline', 'user_timezone', 'airline_sentiment']).size().unstack(fill_value=0)
  min_tweets_timezone_airline = 10
  airline_timezone_sentiment_filtered = airline_timezone_sentiment[airline_timezone_sentiment.sum(axis=1) >= min_tweets_timezone_airline]
  airline_timezone_sentiment_proportions = airline_timezone_sentiment_filtered.apply(lambda x: x / x.sum() * 100, axis=1)

  for airline_to_analyze in airline_timezone_sentiment_proportions.index.get_level_values('airline').unique():
      print(f"\nAnalyzing sentiment performance for {airline_to_analyze} in specific timezones:")

      airline_data_tz = airline_timezone_sentiment_proportions.loc[airline_to_analyze]

      # Timezones with highest positive sentiment proportion for the airline
      top_positive_timezones = airline_data_tz.sort_values(by='positive', ascending=False).head(10)
      print(f"\nTop 10 timezones for {airline_to_analyze} with highest positive sentiment proportion:")
      print(top_positive_timezones[['positive', 'negative', 'neutral']])

      # Timezones with highest negative sentiment proportion for the airline
      top_negative_timezones = airline_data_tz.sort_values(by='negative', ascending=False).head(10)
      print(f"\nTop 10 timezones for {airline_to_analyze} with highest negative sentiment proportion:")
      print(top_negative_timezones[['positive', 'negative', 'neutral']])

      if not top_positive_timezones.empty:
          plt.figure(figsize=(15, 8))
          top_positive_timezones[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('viridis'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Timezones (by Positive Sentiment)')
          plt.xlabel('User Timezone')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()

      if not top_negative_timezones.empty:
          plt.figure(figsize=(15, 8))
          top_negative_timezones[['positive', 'negative', 'neutral']].plot(kind='bar', stacked=True, ax=plt.gca(), color = sns.color_palette('plasma'))
          plt.title(f'Sentiment Distribution for {airline_to_analyze} in Top 10 Timezones (by Negative Sentiment)')
          plt.xlabel('User Timezone')
          plt.ylabel('Proportion (%)')
          plt.xticks(rotation=90)
          plt.legend(title='Sentiment')
          plt.tight_layout()
          plt.show()
```
![image](https://github.com/user-attachments/assets/d362c39e-7de8-4e58-ac97-0c39958a6cfc)
![image](https://github.com/user-attachments/assets/dfffa998-fe50-4a60-a7b4-15d9f5de0ace)
![image](https://github.com/user-attachments/assets/a6ce748e-fa85-4d8b-8850-644ff0cae14b)


![image](https://github.com/user-attachments/assets/29cfa98f-bc4f-4964-8bf9-916d7c73a0f3)
![image](https://github.com/user-attachments/assets/65488fb7-a2ee-47d4-8fc5-64c3fd45a5ea)
![image](https://github.com/user-attachments/assets/d71dd0c8-7492-4bdb-b495-7c910b3d7e80)

![image](https://github.com/user-attachments/assets/e102116f-b8d7-4e89-a20a-5b2840d1e52c)
![image](https://github.com/user-attachments/assets/80243571-9056-4657-bebf-60f7ebd541b4)
![image](https://github.com/user-attachments/assets/48c767bd-43e9-4356-b38c-854e00612a32)

![image](https://github.com/user-attachments/assets/6f1d305a-1a3a-4613-bfe0-565703ce2711)
![image](https://github.com/user-attachments/assets/9ec159a2-2262-461a-906f-b0ad4511bf2a)
![image](https://github.com/user-attachments/assets/ab4c4a37-4865-405d-a10b-684ac387ca22)

![image](https://github.com/user-attachments/assets/53125458-f991-4e41-8383-5a010eaec219)
![image](https://github.com/user-attachments/assets/7f94400e-bbaf-4e88-a45a-b5aeda9b0b60)
![image](https://github.com/user-attachments/assets/b21a2f5a-a2e1-48c3-b1cd-8e370102ef4f)

![image](https://github.com/user-attachments/assets/a2266060-c4a0-46dc-97df-928629847c23)
![image](https://github.com/user-attachments/assets/1eab1284-88ac-442f-87b6-9cd9d50d2fad)
![image](https://github.com/user-attachments/assets/02a7a73d-9bc1-419a-b04c-a94f411a1cb1)
``` python
  timezone_tweet_volume = df['user_timezone'].value_counts().reset_index()
  timezone_tweet_volume.columns = ['user_timezone', 'tweet_volume']

  # Filter out 'No Timezone' if you don't want to include it in the visualization
  timezone_tweet_volume_filtered = timezone_tweet_volume[timezone_tweet_volume['user_timezone'] != 'No Timezone']

  top_n_timezones_volume = 20
  plt.figure(figsize=(15, 8))
  sns.barplot(x='user_timezone', y='tweet_volume', data=timezone_tweet_volume_filtered.head(top_n_timezones_volume), palette='viridis')
  plt.title(f'Tweet Volume by User Timezone (Top {top_n_timezones_volume})')
  plt.xlabel('User Timezone')
  plt.ylabel('Number of Tweets')
  plt.xticks(rotation=90)
  plt.tight_layout()
  plt.show()

  timezone_sentiment_counts = df.groupby(['user_timezone', 'airline_sentiment']).size().unstack(fill_value=0)
  min_tweets_for_sentiment_proportion = 100 # Adjust the threshold how much ever you need
  timezone_sentiment_filtered_for_proportion = timezone_sentiment_counts[timezone_sentiment_counts.sum(axis=1) >= min_tweets_for_sentiment_proportion]
  timezone_sentiment_proportions = timezone_sentiment_filtered_for_proportion.apply(lambda x: x / x.sum() * 100, axis=1)

  top_n_timezones_sentiment = 15 # display of sentiment for top timezones by volume or based on filtered list size

  plt.figure(figsize=(15, 8))
  timezone_sentiment_proportions.head(top_n_timezones_sentiment)[['positive', 'neutral', 'negative']].plot(kind='bar', stacked=True, ax=plt.gca(), color=sns.color_palette('RdYlGn'))
  plt.title(f'Sentiment Distribution by User Timezone (Top {top_n_timezones_sentiment} Timezones by Tweet Volume)')
  plt.xlabel('User Timezone')
  plt.ylabel('Proportion (%)')
  plt.xticks(rotation=90)
  plt.legend(title='Sentiment')
  plt.tight_layout()
  plt.show()

  print("\nSentiment Proportion by User Timezone (filtered for timezones with >= {} tweets):".format(min_tweets_for_sentiment_proportion))
  print(timezone_sentiment_proportions.sort_values(by='negative', ascending=False).head()) # Timezones with highest negative proportion
  print(timezone_sentiment_proportions.sort_values(by='positive', ascending=False).head()) # Timezones with highest positive proportion
```
![image](https://github.com/user-attachments/assets/c8bb2126-ce71-4b3d-8efa-39029afd7965)
![image](https://github.com/user-attachments/assets/d1cb9303-f29c-4bc4-bb11-8a456c26a140)
![image](https://github.com/user-attachments/assets/21ba21f1-f35c-4ef9-8922-4a4163b381c5)

3. **Program Effectiveness & Customer Behavior**
``` python
  tweet_retweet_sentiment = df.groupby('airline_sentiment')['retweet_count'].mean().reset_index()
  print("\nAverage Retweet Count by Sentiment:")
  print(tweet_retweet_sentiment)

  plt.figure(figsize=(8, 6))
  sns.barplot(x='airline_sentiment', y='retweet_count', data=tweet_retweet_sentiment, palette='viridis')
  plt.title('Average Retweet Count by Sentiment')
  plt.xlabel('Sentiment')
  plt.ylabel('Average Retweet Count')
  plt.show()
```
![image](https://github.com/user-attachments/assets/a5ecbbc2-36a2-45ac-920c-5f7ce3495628)
``` python
  sentiment_counts = df['airline_sentiment'].value_counts()
  plt.figure(figsize=(8, 8))
  plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#FFCC17', '#F0561D', '#0066CC'])
  plt.title('Distribution of Airline Sentiment')
  plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
  plt.show()
```
![image](https://github.com/user-attachments/assets/ddcf6e21-e921-4051-acbb-a8cdb4c80507)
``` python
  df['tweet_created'] = pd.to_datetime(df['tweet_created'])
  negative_tweets_time = df[df['airline_sentiment'] == 'negative'].copy()

  # Analyze negative tweets by day of the week
  negative_tweets_time['day_of_week'] = negative_tweets_time['tweet_created'].dt.day_name()
  negative_tweets_by_day = negative_tweets_time['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  print("\nNegative tweets count by day of the week:")
  print(negative_tweets_by_day)

  plt.figure(figsize=(10, 6))
  sns.barplot(x=negative_tweets_by_day.index, y=negative_tweets_by_day.values, palette='viridis')
  plt.title('Negative Tweet Volume by Day of the Week')
  plt.xlabel('Day of the Week')
  plt.ylabel('Number of Negative Tweets')
  plt.show()

  # Analyze negative tweets by hour of the day (using the hour in UTC as the original data seems to be UTC)
  negative_tweets_time['hour_of_day'] = negative_tweets_time['tweet_created'].dt.hour

  negative_tweets_by_hour = negative_tweets_time['hour_of_day'].value_counts().sort_index()

  print("\nNegative tweets count by hour of the day:")
  print(negative_tweets_by_hour)
  plt.figure(figsize=(12, 6))
  sns.lineplot(x=negative_tweets_by_hour.index, y=negative_tweets_by_hour.values)
  plt.title('Negative Tweet Volume by Hour of the Day (UTC)')
  plt.xlabel('Hour of the Day (UTC)')
  plt.ylabel('Number of Negative Tweets')
  plt.xticks(range(0, 24))
  plt.grid(True)
  plt.show()

  # To see if the _proportion_ of negative tweets changes by time, calculating total tweets by time period as well
  all_tweets_time = df.copy()

  all_tweets_time['day_of_week'] = all_tweets_time['tweet_created'].dt.day_name()

  all_tweets_by_day = all_tweets_time['day_of_week'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  all_tweets_time['hour_of_day'] = all_tweets_time['tweet_created'].dt.hour

  all_tweets_by_hour = all_tweets_time['hour_of_day'].value_counts().sort_index()

  negative_proportion_by_day = (negative_tweets_by_day / all_tweets_by_day).dropna()
  print("\nProportion of negative tweets by day of the week:")
  print(negative_proportion_by_day)
  plt.figure(figsize=(12, 6))
  sns.barplot(x=negative_proportion_by_day.index, y=negative_proportion_by_day.values, palette='plasma')
  plt.title('Proportion of Negative Tweets by Day of the Week')
  plt.xlabel('Day of the Week')
  plt.ylabel('Proportion of Negative Tweets')
  plt.show()

  negative_proportion_by_hour = (negative_tweets_by_hour / all_tweets_by_hour).dropna()
  print("\nProportion of negative tweets by hour of the day:")
  print(negative_proportion_by_hour)

  plt.figure(figsize=(12, 6))
  sns.lineplot(x=negative_proportion_by_hour.index, y=negative_proportion_by_hour.values)
  plt.title('Proportion of Negative Tweets by Hour of the Day (UTC)')
  plt.xlabel('Hour of the Day (UTC)')
  plt.ylabel('Proportion of Negative Tweets')
  plt.xticks(range(0, 24))
  plt.grid(True)
  plt.show()
```
![image](https://github.com/user-attachments/assets/9592ccbd-244d-494b-813d-0eeb394c97aa)
![image](https://github.com/user-attachments/assets/2f6b9f38-ae8b-4a03-87ee-d34ef480e345)

![image](https://github.com/user-attachments/assets/c7d73d09-d755-4616-8e86-65d09dab033f)
![image](https://github.com/user-attachments/assets/9f2f051b-0b89-4f9e-80ef-ecd769698e2c)

![image](https://github.com/user-attachments/assets/1af8a24f-9213-46bc-9ea3-f13844becf3f)
![image](https://github.com/user-attachments/assets/39842b5c-1bba-43b0-8357-6f1ea31ac8ad)

![image](https://github.com/user-attachments/assets/e96e2ee0-00ee-4612-bcc1-03c2c70e8420)
![image](https://github.com/user-attachments/assets/8175e8a4-89fb-49fb-b3ef-3bbe8dd297fd)

### Hypothesis testing:
1). Customer Loyalty & Retention
- What are the most frequently cited negativereasons across all airlines?
``` python
  # Null Hypothesis (H0): The frequency of negativereasons is uniformly distributed across all possible negative reasons.
  # Alternative Hypothesis (H1): The frequency of negativereasons is not uniformly distributed, with certain reasons being cited significantly more often than others.

  from scipy.stats import chi2_contingency
  from scipy.stats import chisquare

  # Chi-Squared Test for Uniform Distribution of Negative Reasons
  observed_frequencies = df['negativereason'].value_counts()
  observed_table = pd.DataFrame({'observed': observed_frequencies}).T
  total_negative_reasons = observed_frequencies.sum()
  num_unique_reasons = len(observed_frequencies)

  expected_frequency_per_reason = total_negative_reasons / num_unique_reasons
  expected_frequencies = np.full(num_unique_reasons, expected_frequency_per_reason)

  chi2_stat, p_value = chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)


  print("\nChi-Squared Test for Uniform Distribution of Negative Reasons")
  print(f"Observed Frequencies:\n{observed_frequencies}")
  print(f"Expected Frequency (under H0): {expected_frequency_per_reason:.2f} for each reason")
  print('\n')
  print(f"Chi-squared statistic: {chi2_stat:.4f}")
  print(f"P-value: {p_value:.4f}")

  alpha = 0.05
   if p_value < alpha:
      print(f"\nConclusion: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis.")
      print("There is sufficient evidence to suggest that the frequency of negative reasons is not uniformly distributed.")
   else:
      print(f"\nConclusion: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis.")
      print("There is not enough evidence to suggest that the frequency of negative reasons is not uniformly distributed. The observed distribution is consistent with a uniform distribution.")
```
![image](https://github.com/user-attachments/assets/b59cfacc-1c7d-4690-8874-0456038101b9)
- Which airlines receive the highest proportion of negative sentiment tweets, and which receive the most positive?
``` python
  # Null Hypothesis (H0): There is no significant difference in the proportion of negative (or positive) sentiment tweets across different airlines.
  # Alternative Hypothesis (H1): There is a significant difference in the proportion of negative (or positive) sentiment tweets among different airlines.

  contingency_table = pd.crosstab(df['airline'], df['airline_sentiment'])
  print("\nContingency Table (Airline vs. Sentiment):")
  print(contingency_table)
  chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

  print("\nChi-Squared Test for Independence (Airline vs. Sentiment)")
  print(f"Chi-squared statistic: {chi2_stat:.4f}")
  print(f"P-value: {p_value:.4f}")
  print(f"Degrees of Freedom: {dof}")

  alpha = 0.05  
  if p_value < alpha:
     print(f"\nConclusion: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
     print("There is sufficient evidence to suggest that there is a significant difference in the proportion of sentiment tweets across different airlines.")
  else:
     print(f"\nConclusion: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
     print("There is not enough evidence to suggest a significant difference in the proportion of sentiment tweets across different airlines. The observed distribution is consistent with independence between    airline and sentiment.")
```
![image](https://github.com/user-attachments/assets/8e3f2c36-5c29-430b-880e-9440ec4f72af)
- Is there a correlation between the airline_sentiment_confidence and the likelihood of a tweet being negative?
``` python
  # Null Hypothesis (H0): There is no statistical correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.
  # Alternative Hypothesis (H1): There is a statistical correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.
  from scipy.stats import pearsonr
  correlation, p_value_correlation = pearsonr(df['airline_sentiment_confidence'], df['is_negative'])

  print("\nFormal Test for Correlation between Airline Sentiment Confidence and Likelihood of being Negative")
  print(f"Pearson correlation coefficient: {correlation:.4f}")
  print(f"P-value for the correlation test: {p_value_correlation:.4f}")

  alpha = 0.05
  if p_value_correlation < alpha:
     print(f"\nConclusion: With a p-value of {p_value_correlation:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
     print("There is sufficient evidence to suggest a statistically significant correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.")
  else:
    print(f"\nConclusion: With a p-value of {p_value_correlation:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
    print("There is not enough evidence to suggest a statistically significant correlation between airline_sentiment_confidence and the likelihood of a tweet being negative.")

  if abs(correlation) >= 0.5:
      strength = "strong"
  elif abs(correlation) >= 0.3:
      strength = "moderate"
  elif abs(correlation) >= 0.1:
      strength = "weak"
  else:
     strength = "very weak or no"

  direction = "positive" if correlation > 0 else "negative" if correlation < 0 else "no"
  print(f"The correlation coefficient ({correlation:.4f}) indicates a {strength} {direction} linear relationship.")
```
![image](https://github.com/user-attachments/assets/fb0303b9-37f2-4924-956f-0760aa78531f)

2. Demographic & Geographic Analysis
- Are there specific tweet_locations or user_timezones that show a higher concentration of negative or positive sentiment tweets for particular airlines?
``` python
  # Null Hypothesis (H0): The distribution of sentiment (negative/positive) for a given airline is independent of tweet_location and user_timezone.
  # Alternative Hypothesis (H1): The distribution of sentiment (negative/positive) for a given airline is dependent on tweet_location or user_timezone, indicating a higher concentration of specific sentiments in   certain areas/timezones.

  min_combined_tweets = 100
  for airline_name in df['airline'].unique():
  print(f"\nTesting for {airline_name}")
  airline_df = df[df['airline'] == airline_name].copy()
  airline_df['location_timezone'] = airline_df['tweet_location'] + ' | ' + airline_df['user_timezone']
  contingency_table_combined = pd.crosstab(airline_df['location_timezone'], airline_df['airline_sentiment'])
  contingency_table_filtered = contingency_table_combined[contingency_table_combined.sum(axis=1) >= min_combined_tweets]

  if not contingency_table_filtered.empty and contingency_table_filtered.shape[0] > 1 and contingency_table_filtered.shape[1] > 1:

    print(f"\nPerforming Chi-Squared test for {airline_name} (filtered for location-timezone combinations with >= {min_combined_tweets} tweets)")
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table_filtered)

    print(f"Chi-squared statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p_value < alpha:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print(f"There is sufficient evidence to suggest that the distribution of sentiment for {airline_name} is dependent on tweet_location or user_timezone.")
  else:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print(f"There is not enough evidence to suggest that the distribution of sentiment for {airline_name} is dependent on tweet_location or user_timezone. The observed distribution is consistent with independence.")
    else:
       print(f"\nNot enough data for {airline_name} to perform a reliable Chi-Squared test on the combined location/timezone and sentiment relationship with the applied filter.")
       print(f"Filtered table shape: {contingency_table_filtered.shape}")
```
![image](https://github.com/user-attachments/assets/05bb6a6f-ec5d-4b88-87f8-030a8ed8d114)
- Do the common negativereasons vary significantly by geographic region or user timezone?
``` python
  # Null Hypothesis (H0): The distribution of sentiment (negative/positive) for a given airline is independent of tweet_location and user_timezone.
  # Alternative Hypothesis (H1): The distribution of sentiment (negative/positive) for a given airline is dependent on tweet_location or user_timezone, indicating a higher concentration of specific sentiments in   certain areas/timezones.

  negative_df = df[df['airline_sentiment'] == 'negative'].copy()
  min_timezone_negative_tweets = 50
  min_negativereason_count = 20

  timezone_reason_contingency = pd.crosstab(negative_df['user_timezone'], negative_df['negativereason'])
  timezone_reason_contingency_filtered_tz = timezone_reason_contingency[timezone_reason_contingency.sum(axis=1) >= min_timezone_negative_tweets]

  timezone_reason_contingency_filtered = timezone_reason_contingency_filtered_tz.loc[:, timezone_reason_contingency_filtered_tz.sum(axis=0) >= min_negativereason_count]

  print("\nContingency Table (User Timezone vs. Negative Reason - filtered):")
  print(timezone_reason_contingency_filtered.head()) # Print head as the table can be large

  if not timezone_reason_contingency_filtered.empty and timezone_reason_contingency_filtered.shape[0] > 1 and timezone_reason_contingency_filtered.shape[1] > 1:
    print("\nPerforming Chi-Squared Test for Independence (User Timezone vs. Negative Reason)")
    chi2_stat, p_value, dof, expected = chi2_contingency(timezone_reason_contingency_filtered)

    print(f"Chi-squared statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p_value < alpha:
       print(f"\nConclusion: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
       print("There is sufficient evidence to suggest that the distribution of common negative reasons varies significantly by user_timezone.")
    else:
       print(f"\nConclusion: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
       print("There is not enough evidence to suggest that the distribution of common negative reasons varies significantly by user_timezone. The observed distribution is consistent with independence.")
  else:
     print("\nNot enough data in the filtered contingency table (User Timezone vs. Negative Reason) to perform a reliable Chi-Squared test.")
     print(f"Filtered table shape: {timezone_reason_contingency_filtered.shape}")
```
![image](https://github.com/user-attachments/assets/fc1c63f7-143d-49ad-9fff-ea322294b60a)
- Which airlines demonstrate stronger or weaker sentiment performance in specific geographic areas?
``` python
  min_location_tweets = 50
  print("\nTesting Sentiment Performance across Geographic Areas (Tweet Location) for each Airline")

  for airline_name in df['airline'].unique():
    print(f"\nAnalyzing Sentiment Performance for {airline_name} across Tweet Locations:")
    airline_df = df[df['airline'] == airline_name].copy()
    location_sentiment_contingency = pd.crosstab(airline_df['tweet_location'], airline_df['airline_sentiment'])
    location_sentiment_contingency_filtered = location_sentiment_contingency[location_sentiment_contingency.sum(axis=1) >= min_location_tweets]
    if not location_sentiment_contingency_filtered.empty and location_sentiment_contingency_filtered.shape[0] > 1 and location_sentiment_contingency_filtered.shape[1] > 1:

    print(f"Performing Chi-Squared test for {airline_name} (filtered for locations with >= {min_location_tweets} tweets)")
    chi2_stat, p_value, dof, expected = chi2_contingency(location_sentiment_contingency_filtered)

    print(f"Chi-squared statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")

    alpha = 0.05
    if p_value < alpha:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print(f"There is sufficient evidence to suggest that the sentiment performance for {airline_name} varies significantly across different tweet_locations.")
    else:
      print(f"\nConclusion for {airline_name}: With a p-value of {p_value:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print(f"There is not enough evidence to suggest a significant variation in sentiment performance for {airline_name} across different tweet_locations. The observed distribution is consistent with sentiment        performance being independent of location.")
  else:
    print(f"Not enough data for {airline_name} in the filtered contingency table (Tweet Location vs. Sentiment) to perform a reliable Chi-Squared test.")
    print(f"Filtered table shape: {location_sentiment_contingency_filtered.shape}")
```
![image](https://github.com/user-attachments/assets/9161c790-0dc5-4604-9742-018cf6827840)
- How does the volume of tweets and the sentiment distribution differ across various user_timezones?
``` python
  min_tweets_for_chi2 = 100
  timezone_sentiment_contingency = pd.crosstab(df['user_timezone'], df['airline_sentiment'])
  timezone_sentiment_contingency_filtered = timezone_sentiment_contingency[timezone_sentiment_contingency.sum(axis=1) >= min_tweets_for_chi2]
  print("\nContingency Table (User Timezone vs. Sentiment - filtered for timezones with >= {} tweets):".format(min_tweets_for_chi2))
  print(timezone_sentiment_contingency_filtered.head())

  # Performing the Chi-Squared Test for Independence
  if not timezone_sentiment_contingency_filtered.empty and timezone_sentiment_contingency_filtered.shape[0] > 1 and timezone_sentiment_contingency_filtered.shape[1] > 1:
    chi2_stat_sentiment, p_value_sentiment, dof_sentiment, expected_sentiment = chi2_contingency(timezone_sentiment_contingency_filtered)

    print("\nChi-Squared Test for Independence (User Timezone vs. Sentiment Distribution)")
    print(f"Chi-squared statistic: {chi2_stat_sentiment:.4f}")
    print(f"P-value: {p_value_sentiment:.4f}")
    print(f"Degrees of Freedom: {dof_sentiment}")

    alpha = 0.05
    if p_value_sentiment < alpha:
      print(f"\nConclusion: With a p-value of {p_value_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
      print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across user timezones.")
    else:
      print(f"\nConclusion: With a p-value of {p_value_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
      print("There is not enough evidence to suggest a significant difference in sentiment distribution across user timezones. The observed distribution is consistent with sentiment distribution being   independent of timezone.")
  else:
    print("\nNot enough data in the filtered contingency table (User Timezone vs. Sentiment) to perform a reliable Chi-Squared test.")
    print(f"Filtered table shape: {timezone_sentiment_contingency_filtered.shape}")

  # Test 2: Chi-Squared Test for Uniform Distribution of Tweet Volume across Timezones (considering only significant timezones)
  timezone_tweet_volume_filtered = timezone_sentiment_contingency_filtered.sum(axis=1)

  if not timezone_tweet_volume_filtered.empty and len(timezone_tweet_volume_filtered) > 1:
    observed_tweet_volumes = timezone_tweet_volume_filtered.values
    total_volume_filtered = observed_tweet_volumes.sum()
    num_timezones_filtered = len(observed_tweet_volumes)
    expected_volume_per_timezone = total_volume_filtered / num_timezones_filtered
    expected_tweet_volumes = np.full(num_timezones_filtered, expected_volume_per_timezone)

    # Perform the Chi-Squared Test for Uniformity
    chi2_stat_volume, p_value_volume = chisquare(f_obs=observed_tweet_volumes, f_exp=expected_tweet_volumes)

    print("\nChi-Squared Test for Uniform Distribution of Tweet Volume across Filtered User Timezones")
    print(f"Observed Tweet Volumes:\n{timezone_tweet_volume_filtered.head()}") # Print head as this can be long
    print(f"Expected Tweet Volume (under H0): {expected_volume_per_timezone:.2f} for each timezone")
    print('\n')
    print(f"Chi-squared statistic: {chi2_stat_volume:.4f}")
    print(f"P-value: {p_value_volume:.4f}")
    alpha = 0.05
    if p_value_volume < alpha:
      print(f"\nConclusion: With a p-value of {p_value_volume:.4f} (less than alpha={alpha}), we reject the null hypothesis.")
      print("There is sufficient evidence to suggest that the tweet volume is not uniformly distributed across the filtered user timezones.")
    else:
      print(f"\nConclusion: With a p-value of {p_value_volume:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis.")
      print("There is not enough evidence to suggest that the tweet volume is not uniformly distributed across the filtered user timezones. The observed distribution is consistent with a uniform distribution.")

  else:
      print("\nNot enough data in the filtered list of timezones to perform a reliable Chi-Squared test for uniform volume.")
      print(f"Number of filtered timezones: {len(timezone_tweet_volume_filtered)}")

  print("\nOverall Hypothesis Test Conclusion (Timezone vs. Tweet Volume and Sentiment)")
  alpha = 0.05
  if p_value_sentiment < alpha or p_value_volume < alpha:
      print("Based on the Chi-Squared tests for sentiment distribution and tweet volume, we reject the Null Hypothesis (H0).")
      print("There is significant evidence to suggest that the volume of tweets and/or the sentiment distribution differ across user_timezones.")
  else:
      print("Based on the Chi-Squared tests, we fail to reject the Null Hypothesis (H0).")
      print("There is not enough evidence to suggest that the volume of tweets or the sentiment distribution differ significantly across user_timezones.")
```
![image](https://github.com/user-attachments/assets/45742bdc-b240-4000-9709-b69a5141c2db)
![image](https://github.com/user-attachments/assets/b28d7efe-ebab-4d3c-abf9-09a511ca738d)

3. Program Effectiveness & Customer Behavior
How does the retweet_count differ for tweets with positive, neutral, and negative sentiments?
``` python
  # Null Hypothesis (H0): There is no significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.
  # Alternative Hypothesis (H1): There is a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.

  import statsmodels.api as sm
  from statsmodels.formula.api import ols
  from statsmodels.stats.multicomp import pairwise_tukeyhsd

  model = ols('retweet_count ~ C(airline_sentiment)', data=df).fit()
  anova_table = sm.stats.anova_lm(model, typ=2) # typ=2 for unbalanced data
  print("\nANOVA Test for Retweet Count by Sentiment:")
  print(anova_table)

  alpha = 0.05
  p_value_anova = anova_table['PR(>F)'][0]
  if p_value_anova < alpha:
    print(f"\nConclusion: With a p-value of {p_value_anova:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
    print("There is sufficient evidence to suggest that there is a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.")

    # Perform Tukey's HSD post-hoc test to see which pairs of sentiments differ
    print("\nPerforming Tukey's HSD Post-Hoc Test:")
    tukey_result = pairwise_tukeyhsd(endog=df['retweet_count'], groups=df['airline_sentiment'], alpha=alpha)
    print(tukey_result)
    print("\nInterpretation of Tukey's HSD:")
    print("The 'reject' column indicates if the difference between the means of the two groups (group1 vs group2) is statistically significant (True means significant difference).")
  else:
    print(f"\nConclusion: With a p-value of {p_value_anova:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
    print("There is not enough evidence to suggest a significant difference in the mean retweet_count among tweets with positive, neutral, and negative sentiments.")
    correlation_hour_retweet, p_value_hour_retweet = pearsonr(df['tweet_created'].dt.hour, df['retweet_count'])

    print("\nPearson Correlation between Hour of Day (UTC) and Retweet Count")
    print(f"Correlation coefficient: {correlation_hour_retweet:.4f}")
    print(f"P-value: {p_value_hour_retweet:.4f}")

  alpha = 0.05
  if p_value_hour_retweet < alpha:
      print("Conclusion: Significant linear correlation between hour of day and retweet count.")
  else:
      print("Conclusion: No significant linear correlation between hour of day and retweet count.")

  # For day of the week and retweet count (ANOVA)
  if not df.empty:
      df['day_of_week_num'] = df['tweet_created'].dt.dayofweek # Monday=0, Sunday=6
      day_anova_model = ols('retweet_count ~ C(day_of_week_num)', data=df).fit()
      day_anova_table = sm.stats.anova_lm(day_anova_model, typ=2)

      print("\nANOVA Test for Retweet Count by Day of the Week:")
      print(day_anova_table)

      p_value_day_anova = day_anova_table['PR(>F)'][0]

      if p_value_day_anova < alpha:
          print(f"\nConclusion: With a p-value of {p_value_day_anova:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that there is a significant difference in the mean retweet_count across different days of the week.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_day_anova:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in the mean retweet_count across different days of the week.")
  else:
      print("DataFrame is empty, cannot perform ANOVA by Day of the Week.")

  # For timing of tweets and sentiment distribution (Chi-Squared)
  # Day of the week vs. Sentiment
  contingency_table_day_sentiment = pd.crosstab(df['tweet_created'].dt.day_name(), df['airline_sentiment'])
  contingency_table_day_sentiment = contingency_table_day_sentiment.reindex(index=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

  if not contingency_table_day_sentiment.empty and contingency_table_day_sentiment.shape[0] > 1 and contingency_table_day_sentiment.shape[1] > 1:
      chi2_stat_day_sentiment, p_value_day_sentiment, dof_day_sentiment, expected_day_sentiment = chi2_contingency(contingency_table_day_sentiment)

      print("\nChi-Squared Test for Independence (Day of Week vs. Sentiment)")
      print(f"Chi-squared statistic: {chi2_stat_day_sentiment:.4f}")
      print(f"P-value: {p_value_day_sentiment:.4f}")
      print(f"Degrees of Freedom: {dof_day_sentiment}")

      alpha = 0.05
      if p_value_day_sentiment < alpha:
          print(f"\nConclusion: With a p-value of {p_value_day_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across different days of the week.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_day_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in sentiment distribution across different days of the week.")
  else:
      print("Not enough data in the contingency table (Day of Week vs. Sentiment) to perform a reliable Chi-Squared test.")
      print(f"Table shape: {contingency_table_day_sentiment.shape}")

  # Hour of the day vs. Sentiment
  contingency_table_hour_sentiment = pd.crosstab(df['tweet_created'].dt.hour, df['airline_sentiment'])
  min_tweets_per_hour = 10
  contingency_table_hour_sentiment_filtered = contingency_table_hour_sentiment[contingency_table_hour_sentiment.sum(axis=1) >= min_tweets_per_hour]


  if not contingency_table_hour_sentiment_filtered.empty and contingency_table_hour_sentiment_filtered.shape[0] > 1 and contingency_table_hour_sentiment_filtered.shape[1] > 1:
      chi2_stat_hour_sentiment, p_value_hour_sentiment, dof_hour_sentiment, expected_hour_sentiment = chi2_contingency(contingency_table_hour_sentiment_filtered)\
      print("\nChi-Squared Test for Independence (Hour of Day vs. Sentiment - filtered)")
      print(f"Chi-squared statistic: {chi2_stat_hour_sentiment:.4f}")
      print(f"P-value: {p_value_hour_sentiment:.4f}")
      print(f"Degrees of Freedom: {dof_hour_sentiment}")

      alpha = 0.05
      if p_value_hour_sentiment < alpha:
          print(f"\nConclusion: With a p-value of {p_value_hour_sentiment:.4f} (less than alpha={alpha}), we reject the null hypothesis (H0).")
          print("There is sufficient evidence to suggest that the sentiment distribution differs significantly across different hours of the day.")
      else:
          print(f"\nConclusion: With a p-value of {p_value_hour_sentiment:.4f} (greater than or equal to alpha={alpha}), we fail to reject the null hypothesis (H0).")
          print("There is not enough evidence to suggest a significant difference in sentiment distribution across different hours of the day.")
  else:
      print("\nNot enough data in the filtered contingency table (Hour of Day vs. Sentiment) to perform a reliable Chi-Squared test.")
      print(f"Filtered table shape: {contingency_table_hour_sentiment_filtered.shape}")
```
![image](https://github.com/user-attachments/assets/17ca8f3b-3c04-4401-967a-483c12b64ee3)
![image](https://github.com/user-attachments/assets/e07cf5d3-cb2e-434f-82a5-74521eb0c249)

- Are there specific days or times (tweet_created) when negative sentiment tweets are more prevalent, suggesting periods of heightened
``` python
  # Null Hypothesis (H0): The proportion of negative sentiment tweets is consistent across different days of the week and times of the day.
  # Alternative Hypothesis (H1): The proportion of negative sentiment tweets is significantly higher on certain days of the week or during specific times of the day.

  print("\nOverall Hypothesis Test Conclusion (Timing of Negative Sentiment)")
  alpha = 0.05
  if p_value_day_sentiment < alpha or p_value_hour_sentiment < alpha:
      print("Based on the Chi-Squared tests, we reject the Null Hypothesis (H0).")
      print("There is sufficient evidence to suggest that the proportion of negative sentiment tweets varies significantly across different days of the week and/or times of the day.")
      print("This supports the Alternative Hypothesis (H1) that negative sentiment is significantly higher on certain days or times.")
  else:
      print("Based on the Chi-Squared tests, we fail to reject the Null Hypothesis (H0).")
      print("There is not enough evidence to suggest that the proportion of negative sentiment tweets varies significantly across different days of the week or times of the day.")
      print("The observed distribution is consistent with the Null Hypothesis (H0) that the proportion is consistent.")
  ```
![image](https://github.com/user-attachments/assets/5648f870-18a0-4802-a2b0-dd1c94f02067)

### Insights
Based on the analysis from Twitter interactions regarding the airlines is mentioned bekow

1. **Customer Loyalty & Retention

- The most negative reasons among the airlines were Miscellaneous reasons, Custoemr Support, flight delays.

 a. Delta, Southwest, United had miscllaneous problems like online cancellation, flight service, food service, longer wait time, exchange of seats for reservation purpose, server issues on the portals for  booking nor cancellation,wait time for bag checks 

 b. American, US Airways, United, Southwest Airlines faced Custoemr Support issues.

 c. Booking problems tooks place with United airways.

 d. American had delayed flights.

- We see that the passengers had faced miscallenous issues, Customer service issues, delay in flights

- The Airline Sentiment Proportion tells us that Virgin America airline had the highest positive tweets. Whereas, United, US Airways, American ailrines had the most negative tweets.

2. **Demographic & Geographic Analysis**

- A large portion of tweets (over 3,000 negative) lack location data, indicating a potential gap for geographical analysis.

- High volumes of negative tweets originate from major U.S. cities like Boston, Chicago, New York (various entries), Los Angeles, and Washington D.C., likely reflecting high travel volumes.

-  "No Timezone": Similar to location, a significant number of tweets (over 3,000 negative) are missing timezone information, impacting time-based analysis.

- "Eastern Time (US & Canada)" accounts for the highest volume of tweets across all sentiments, as expected due to its population density.

- Central Time (US & Canada)" and "Pacific Time (US & Canada)" also show substantial tweet volumes, predominantly negative.

- Timezones like "Quito," "London," "Amsterdam," and "Sydney" indicate the global reach of the tweets, with "Quito" showing a surprisingly high number of negative tweets.

- The proportion of negative sentiment remains high across diverse locations. Many cities, including Washington DC, NYC, Brooklyn, NY, San Francisco, and Chicago, show negative sentiment proportions above 65%, with some even exceeding 70%.

- "No location" tweets show a substantial negative sentiment (around 66%), reinforcing the idea that a large segment of the data lacks specific geographical context but still expresses negative feedback.

- While some locations like Dallas, TX, show a relatively lower proportion of negative sentiment (46.29%), with higher neutral and positive proportions compared to other cities. This could indicate regional differences in service quality or customer expectations.

- Quito: Stands out with a very high proportion of negative sentiment (70.8%), suggesting a particularly challenging experience for users in this timezone.

- Amsterdam: Shows a significantly lower proportion of negative sentiment (48.6%) and a much higher proportion of neutral sentiment (35.1%), making it an outlier compared to other timezones. This might indicate better service or different tweeting habits.

- Hawaii: Also exhibits a relatively higher neutral proportion (25%) compared to many other timezones.

- Distribution of Negative Reasons by Tweet Location are:

1. Chicago - Bad flights, Can't tell.(Personal issue's, Hygeiene issue's, Mannerism, food, travel safetiness and wwellness).

2. Austin, TX - Cancelled flights, Customer Support Issue, Flight Booking problems, Other Issues.

3. Boston, MA - Flight Attendant complaints, Lost Luggage, other issue's, Late flights.

4.  Brooklyn, NY - Late flights, Longlines.

- Distribution of Negative Reasons by User Timezone are:

1. Alaska - Can't tell, Cancelled flights, Customer Support Issue, Flight Booking problems, Late flights.

2. Atlantic time - Lost Luggage, Long lines, Late flights 

3. Arizona - Bad flights.

4. Amsterdam - Other issue's.

- In the above stats, we see that the other's and Customer support issue's were the main concerns where the passengers have been facing across all locations and time zones.

3. Program Effectiveness & Customer Behavior

- Around 62% were most of the negative tweets across the airlines.

- The peak time of tweets was in the morning and night.

- Sunday ranks the most number of tweets in the days of the week as well as hourly basis and the least numebr of tweets are on Wednesday and Thursday.

### Recommendations
**1. Enhancing Customer Loyalty & Retention**

- Address Core Negative Drivers: Implement focused strategies to mitigate the impact of "Miscellaneous reasons," "Customer Support issues," and "Flight delays," as these are the most prevalent sources of negative sentiment.

- Deconstruct "Miscellaneous" Issues: Conduct deeper qualitative analysis (e.g., text mining, manual review) on tweets categorized as "Miscellaneous" to identify specific, recurring underlying problems (e.g., online cancellation difficulties, food service, seat exchange issues, portal server problems, long wait times for bag checks). Once identified, create new actionable categories and develop targeted solutions.

- Elevate Customer Support:

  - Targeted Training: Provide enhanced training for customer service teams,  particularly for American, US Airways, United, and Southwest Airlines, focusing on empathy, efficient problem resolution, and communication skills.

  - Resource Allocation: Increase staffing or optimize scheduling for customer support channels (phone, chat, social media) to reduce response times and improve resolution rates.

- Improve Operational Reliability:

  - Flight Delay Reduction: American Airlines should specifically focus on initiatives to reduce flight delays, such as optimizing scheduling, improving maintenance turnaround times, and enhancing ground operations efficiency.

  - Booking System Optimization: United Airways must prioritize resolving "Booking problems" by improving the user experience and reliability of their online booking platforms.

- Learn from Best Practices: Investigate the factors contributing to Virgin America's highest positive tweet proportion. Analyze their operational, communication, and customer service strategies to identify transferable best practices.

**2. Leveraging Demographic & Geographic Insights**

- Enhance Data Granularity:
  - Incentivize Location/Timezone Sharing: Explore ways to encourage users to enable location/timezone sharing on their tweets (if privacy policies permit) to enrich the dataset for more precise geographical and temporal analysis.

  - Geolocation Inference: Implement or utilize tools to infer location/timezone data where explicitly missing, to gain a more complete picture of regional issues.

- Implement Regionalized Customer Service:
  - Staffing & Language: Adjust customer service staffing based on high negative tweet volumes from major U.S. cities (Boston, Chicago, New York, Los Angeles, Washington D.C.) and dominant timezones (Eastern, Central, Pacific). Consider language support for international timezones like Quito.
  - Localized Training: Train customer service agents on prevalent negative reasons specific to their region (e.g., Austin agents on cancellations, Boston on flight attendant complaints/lost luggage, Chicago on "Bad Flights").

- Targeted Operational Improvements (Geographic): Address specific operational issues identified in high-complaint cities (e.g., focus on reducing "Late Flights" and "Longlines" in Brooklyn, "Cancelled Flights" and "Other Issues" in Austin, TX).

- Analyze Positive Outliers: Study the practices or demographics in locations/timezones with lower negative proportions (e.g., Dallas, TX; Amsterdam; Hawaii) to identify successful strategies that could be replicated elsewhere.

- Proactive Localized Communication: Issue timely and localized alerts or updates for known issues (e.g., weather-related delays, operational disruptions) to manage customer expectations and potentially reduce negative tweets from affected regions.

**3. Optimizing Program Effectiveness & Customer Behavior Response**

- Prioritize Negative Tweet Resolution: Given that approximately 62% of tweets are negative, dedicate substantial resources to rapid response and resolution of these complaints. Prompt and effective resolution can mitigate damage, improve individual customer satisfaction, and potentially shift sentiment.

- Align Staffing with Peak Hours: Optimize customer service and social media monitoring team schedules to align with peak tweeting times (morning and night) to ensure immediate engagement with customer concerns.

- Strengthen Weekend Readiness: Allocate additional customer support and operational resources for Sundays, as this day consistently shows the highest volume of tweets. This proactive approach can manage increased passenger activity and potential issues.

- Strategic Communication Timing: Consider the lower tweet volumes on Wednesdays and Thursdays for scheduling non-urgent communications, marketing campaigns, or surveys, as they might achieve higher visibility during these quieter periods.

- Proactive Issue Management: For recurring issues identified (e.g., "Customer Support," "Flight Delays"), develop proactive communication strategies (e.g., automated updates, self-service options) to address concerns before they escalate into public negative tweets.
