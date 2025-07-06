import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

filename=os.path.join(os.path.dirname(__file__),'datasets','Mail.csv')

df =pd.read_csv(filename)

df.describe()

plt.figure(1,figsize=(15,6))
n=0

1
sns.pairplot(df, vars = ['Spending Score (1-100)', 'Annual Income (k$)', 'Age'], hue = "Gender")


