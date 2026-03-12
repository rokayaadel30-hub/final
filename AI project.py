# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# make vsc read the csv
data = pd.read_csv('student_exam_scores.csv')
#show the first five of data
data.head()
#hours_studied vs exam_score(line_plot
df = pd.read_csv('student_exam_scores.csv')
plt.figure(figsize=(8,6))
plt.scatter(df['hours_studied'], df['exam_score'], color='red')
plt.title('hours_studied vs exam_score')
plt.xlabel('hours_studied')
plt.ylabel('exam_score')
plt.show()
#select the data we need to test and train it
x = data[['student_id', 'sleep_hours', 'attendance_percent','previous_scores']]
y = data[['hours_studied','exam_score']]
#Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.2, random_state=29 )
#train the data
model = LinearRegression()
model.fit(x_train,y_train)
#evaluate the performance of model
y_pred = model.predict(x_test)
#Error rate in the model
mse = mean_squared_error(y_test,y_pred)
#Comparison between y_test and y_pred to know ,Are the predictions appropriate or not? 
r2 = r2_score(y_test,y_pred)*100

print(f"Mean_Squared_Error(MSE): {mse}")
print(f"R_squared(R2): {r2}")

