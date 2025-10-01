import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_multiclass_auc(file_path, true_col="CHOICE", pred_col="prediction"):
    """
    Tính chỉ số AUC đa lớp (OVO) từ file CSV.
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn file CSV chứa kết quả.
    true_col : str
        Tên cột nhãn thực tế.
    pred_col : str
        Tên cột nhãn dự đoán.
    
    Returns:
    --------
    dict : chứa giá trị AUC cho OVO
    """
    # Đọc dữ liệu
    df = pd.read_csv(file_path)
    y_true = df[true_col].values
    y_pred = df[pred_col].values
    
    # Xác định tất cả các lớp
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Chuyển sang dạng one-hot
    y_true_bin = label_binarize(y_true, classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)
    
    # Tính AUC
    auc_ovo = roc_auc_score(y_true_bin, y_pred_bin, multi_class="ovo", average="macro")
    
    return auc_ovo

# In result
result = compute_multiclass_auc("results/result4.csv")
print("AUC_OVO (M-metric):", result)