### **Supervised, Semi-Supervised, and Unsupervised Learning with SVMs & Clustering**  
**Author:** Aditya Dutta  
**Date:** Sept 2024 - Oct 2024  
**Technologies:** Python, Scikit-Learn, SMOTE, K-Means, Spectral Clustering, SVM  

---

## **Project Overview**  
This project applies **Supervised, Semi-Supervised, and Unsupervised Learning** methods to classify the **Breast Cancer Wisconsin Dataset** and analyze the **Banknote Authentication Dataset** using **Active Learning with SVMs**. The goal is to compare different machine learning strategies in handling classification and clustering problems.

---

## **Dataset**  
🔹 **Breast Cancer Wisconsin Dataset** ([UCI Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29))  
- **569 instances** with **30 attributes**  
- **Binary classification**: **Benign (B) vs Malignant (M)**  

🔹 **Banknote Authentication Dataset** ([UCI Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication))  
- **1,372 instances** with **4 attributes**  
- **Binary classification**: **Authentic vs Forged Banknotes**  

---

## **Methods & Techniques**  

### **1. Supervised Learning**  
✔ **L1-Penalized SVM** – Used **5-fold cross-validation** to tune penalty parameters  
✔ **Monte Carlo Simulation** – Evaluated models over **30 iterations** for stable results  
✔ **Performance Metrics** – **Accuracy, Precision, Recall, F1-Score, ROC AUC**  

### **2. Semi-Supervised Learning (Self-Training with SVM)**  
📌 **Labeled only 50% of the training set** (randomly selected)  
📌 **Trained SVM on labeled data, iteratively labeled remaining data**  
📌 **Tested final SVM and compared with fully supervised model**  

### **3. Unsupervised Learning (K-Means & Spectral Clustering)**  
🔹 **K-Means Clustering** – Used **majority voting** from **30 closest points** for classification  
🔹 **Spectral Clustering** – Used **RBF Kernel** to capture **non-linear cluster structures**  
🔹 **Monte Carlo Runs (30 iterations)** – Stability analysis  

### **4. Active Learning with SVM**  
📌 **Compared Passive vs. Active Learning on Banknote Authentication Dataset**  
📌 **Trained an SVM incrementally, selecting new samples closest to decision boundary**  
📌 **Plotted learning curve comparing Active & Passive Learning errors**  

---

## **Results & Findings**  

| Model                  | Accuracy | Precision | Recall | F1-Score | ROC AUC |  
|----------------------|------------|------------|------------|------------|------------|  
| **Supervised SVM**   | **98.6%**  | **0.99**  | **0.98**  | **0.98**  | **0.99**  |  
| **Semi-Supervised SVM** | **96.9%**  | **0.97**  | **0.97**  | **0.97**  | **0.99**  |  
| **K-Means Clustering** | **92.5%**  | **0.93**  | **0.92**  | **0.92**  | **0.98**  |  
| **Spectral Clustering** | **94.7%**  | **0.95**  | **0.94**  | **0.94**  | **0.98**  |  
| **Active Learning SVM** | **99.1%**  | **0.99**  | **0.99**  | **0.99**  | **0.99** ✅ |  

✔ **Active Learning improved SVM performance with fewer training samples**  
✔ **K-Means was less accurate but provided valuable clustering insights**  
✔ **Spectral Clustering performed better in non-linear separations**  

---

## **How to Run the Project**  
1️⃣ Install dependencies:  
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```
2️⃣ Download the datasets:  
- **Breast Cancer Wisconsin Dataset** ([Link](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29))  
- **Banknote Authentication Dataset** ([Link](https://archive.ics.uci.edu/ml/datasets/banknote+authentication))  

3️⃣ Run the **Jupyter Notebook** or execute the script:  
```bash
python supervised_semi_unsupervised.py
```

---

## **Key Takeaways**  
🔹 **Supervised learning outperforms semi-supervised and unsupervised approaches**  
🔹 **Active Learning SVM converges faster than Passive Learning**  
🔹 **Spectral Clustering handles non-linearity better than K-Means**  
🔹 **K-Means requires careful centroid initialization to avoid local minima**  

---

## **Future Improvements**  
🚀 **Explore Deep Learning models (CNNs, Autoencoders) for feature extraction**  
🚀 **Test different kernel functions in SVM and Spectral Clustering**  
🚀 **Apply Hierarchical Clustering for better interpretability**  

---

### 📌 **Contact**  
For any questions or improvements, feel free to connect! 🚀  
