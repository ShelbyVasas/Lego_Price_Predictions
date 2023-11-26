
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lego_data = pd.read_csv('./lego_sets.csv')
lego_data

# %%
columns_to_keep = ['Set_ID', 'Name', 'Year', 'Pieces', 'Minifigures', 'Owned', 'USD_MSRP', 'Current_Price']
data = lego_data[columns_to_keep]
data.dropna(inplace=True)
data

# %%
lego_subset = data.sample(500)
sns.pairplot(lego_subset)
plt.show()

corr = lego_subset.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# %% Q1 - Is it possible to accurately calculate the listing price of any given set based on the current price and set size?
# Create features and target variables
X = data[['Pieces', 'Minifigures', 'Current_Price']]
y = data['USD_MSRP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

prediction = model.predict([[1545, 5, 170.15]])
prediction

# %% Q2 Can the number of pieces in a set be used to predict the number of minifigures?
lego_subset = data.sample(500)
sns.pairplot(lego_subset, vars=['Minifigures', 'Pieces'])
plt.show()

X = data[['Pieces']]
y = data['Minifigures']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

prediction = model.predict([[1545]])
prediction2 = model.predict([[387]])
prediction3 = model.predict([[600]])

prediction
# prediction2
# prediction3

## 387 : 3 , 600 : 4
# %%
