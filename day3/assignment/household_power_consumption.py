# %%
import numpy as np
import pandas as pd

# %%
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
#handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
#The prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']
