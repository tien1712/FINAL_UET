import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Đọc dữ liệu
df = pd.read_csv("results/result7.csv")

# Kiểm tra xem có đủ cột không
if not {"prediction", "CHOICE"}.issubset(df.columns):
    raise ValueError("File result5.csv phải có 2 cột: 'prediction' và 'CHOICE'.")

# Kiểm tra và xử lý giá trị NaN
print("Số lượng giá trị NaN trong cột prediction:", df["prediction"].isna().sum())
print("Số lượng giá trị NaN trong cột CHOICE:", df["CHOICE"].isna().sum())
print(df[df["prediction"].isna()])

# Lấy nhãn dự đoán và nhãn thực
y_pred = df["prediction"]
y_true = df["CHOICE"]

# In confusion matrix
print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# In classification report
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, digits=4))