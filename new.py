import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ( accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, recall_score , precision_score)
data = {
    "Hours_Studied": [2.5, 5.1, 3.2, 8.5, 1.5, 9.2, 5.5, 2.0, 6.8, 4.0, 7.5, 3.8],
    "Attendance_Pct": [75, 85, 60, 95, 50, 98, 80, 55, 88, 70, 92, 65],
    "Sleep_Hours": [7, 6, 8, 7, 5, 8, 7, 6, 7, 6, 8, 7],
    "Previous_Score": [55, 70, 50, 88, 40, 92, 65, 45, 78, 60, 82, 58],
    "Final_Score": [58, 78, 52, 92, 42, 96, 74, 48, 84, 62, 88, 60],
}
df=pd.DataFrame(data)
status_summary = pd.DataFrame ( 
    {
       ' Mean' :df.mean() ,
        'Var' : df.var(),
        'Median': df.median() ,
        'Std Dev' : df.std(),
    }
)
print(status_summary)
print('\n')
x = df[["Hours_Studied", "Attendance_Pct", "Sleep_Hours", "Previous_Score"]]
y = df["Final_Score"]
x_train , x_test , y_train , y_test = train_test_split(x,y , test_size=0.2, random_state =42)
scale = StandardScaler()
x_trainscaled= scale.fit_transform(x_train)
x_testscaled= scale.transform(x_test)
model = LinearRegression()
model.fit(x_trainscaled, y_train)
y_pred =model.predict(x_testscaled)
print("==============================================")
print("   2. REGRESSION EVALUATION METRICS")
print("==============================================")
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print('Mean absolute error' , mae)
print('mean squared error' ,mse)
print('r2 score (accuracy)', r2)
print("==============================================")
print("   3. CLASSIFICATION METRICS (PASS / FAIL)")
print("==============================================")
y_test_class =(y_test>= 50).astype(int)
y_pred_class =(y_pred>= 50).astype(int)
acc=accuracy_score(y_test_class , y_pred_class)
prec=precision_score(y_test_class , y_pred_class , zero_division=0)
rec=recall_score(y_test_class , y_pred_class , zero_division=0)
f1=f1_score(y_test_class , y_pred_class , zero_division=0)
print('accuracy' , acc)
print('precision', prec)
print('recall ', rec)
print('f1 score', f1)
print("==============================================")
print("   4. CUSTOM STUDENT PREDICTION")
print("==============================================")
print('enter parameters')

new=np.array([[9.0,70,4,65]])
new_scaled= scale.transform(new)
predicted_score = model.predict(new_scaled)[0]
status = "PASS ✅" if predicted_score >= 50 else "FAIL ❌"

print(f"Inputs          : {new[0].tolist()}")
print(f"Predicted Score : {predicted_score:.2f} / 100")
print(f"Final Outcome   : {status}")
