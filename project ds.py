import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
url = "c:/Users/91778/OneDrive - RGUKT Ongole/Desktop/project/vijay.csv"  
df = pd.read_csv(url)
print(df.info())
df = df.dropna()  
plt.figure(figsize=(10, 6))
sns.histplot(df['NumericColumn'], bins=30, kde=True)
plt.title('Distribution of NumericColumn')
plt.xlabel('NumericColumn')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Feature1', y='Feature2', data=df)
plt.title('Scatter Plot between Feature1 and Feature2')
plt.show()
mean_value = np.mean(df['NumericColumn'])
std_dev = np.std(df['NumericColumn'])
print(f"Mean: {mean_value}, Standard Deviation: {std_dev}")
df['NewFeature'] = df['Feature1'] * df['Feature2']
df['Date'] = pd.to_datetime(df['DateColumn'])
plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='NumericColumn', data=df)
plt.title('Time Series Analysis')
plt.xlabel('Date')
plt.ylabel('NumericColumn')
plt.show()
fig = px.scatter(df, x='Feature1', y='Feature2', color='TargetColumn', size='NumericColumn')
fig.show()
