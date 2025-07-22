import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

filename = os.path.join(os.path.dirname(__file__), 'datasets', 'students_scores.csv')
df = pd.read_csv(filename)


df['Group'] = df.apply(
    lambda row: 1 if (row['Physics'] + row['Math'] + row['Chemistry']) >
                     (row['Marketing'] + row['Business Law'] + row['Economics'] + row['Accounting'])
    else 0,
    axis=1
)

df['Science_Score'] = df['Physics'] + df['Math'] + df['Chemistry']
df['Commerce_Score'] = df['Marketing'] + df['Business Law'] + df['Economics'] + df['Accounting']


X = df[['Science_Score', 'Commerce_Score']]
y = df['Group']


svm_clf = SVC(kernel='linear')
svm_clf.fit(X, y)


plt.figure(figsize=(10, 6))

colors = ['red' if label == 0 else 'blue' for label in y]
plt.scatter(X['Science_Score'], X['Commerce_Score'], c=colors, s=50, edgecolors='k', label='Students')

# Labels and title
plt.xlabel('Science Subjects Total Score (Physics + Math + Chemistry)')
plt.ylabel('Commerce Subjects Total Score (Marketing + Business Law + Economics + Accounting)')
plt.title('Student Group Classification (Science=Blue, Commerce=Red)')


w = svm_clf.coef_[0]
b = svm_clf.intercept_[0]


x_points = np.linspace(X['Science_Score'].min() - 10, X['Science_Score'].max() + 10, 100)
y_points = (w[0] * x_points + b) / w[1]



x_points = np.linspace(X['Science_Score'].min() - 10, X['Science_Score'].max() + 10, 100)
y_points = -(w[0] * x_points + b) / w[1]

plt.plot(x_points, y_points, 'k--', label='Decision Boundary')

plt.legend()
plt.grid(True)
plt.show()
