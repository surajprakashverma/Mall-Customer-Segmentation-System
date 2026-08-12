# 🛍️ Mall Customer Segmentation System

A Flask-based Machine Learning web application that segments mall customers into distinct groups using **K-Means clustering**, based on **Annual Income** and **Spending Score**.

Users enter a new customer's annual income and spending score, and the app instantly identifies which segment they belong to — complete with a business-friendly description and a visual scatter plot showing where the new customer falls among the existing clusters.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KMeans-F7931E.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render-46E3B7.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌐 Live Demo

**🚀 Try it live:**

https://your-render-app.onrender.com

> _Note: Render free-tier services may take a few seconds to wake up after inactivity._

---

## ✨ Features

### 🎯 Customer Segmentation

- K-Means clustering trained on 200 mall customers (Annual Income vs Spending Score).
- Optimal cluster count selected via the **Elbow Method**.
- Instantly predicts which of the 5 segments a new customer belongs to.
- Dynamic, business-friendly segment labels (e.g. "High Income, High Spending") generated from each cluster's centroid — not hardcoded, so it stays accurate regardless of cluster ordering.

### 📊 Visualization

- Live-generated scatter plot showing all existing customer clusters, their centroids, and the new customer's position highlighted with a star marker.
- Plot rendered server-side with Matplotlib and returned as a base64-encoded image — no static files needed.

### 🎨 User Interface

- Modern glassmorphism design.
- Fully responsive two-column layout.
- Animated gradient background with floating shapes.
- Result shown in a scrollable popup modal with segment label, description, and plot.

### ☁️ Deployment Ready

- Render deployment support.
- Gunicorn production server.
- Dataset (`Mall_Customers.csv`) excluded from the deployed instance via `.gitignore`.

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Backend | Flask, Python |
| Machine Learning | scikit-learn (K-Means) |
| Visualization | Matplotlib |
| Data Handling | NumPy |
| Frontend | HTML5, CSS3, JavaScript |
| UI Design | Glassmorphism, Animations |
| Deployment | Render |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```text
MallCustomerSegmentation/
│
├── app.py
├── model.pkl
├── cluster_data.pkl
├── requirements.txt
├── render.yaml
├── README.md
├── .gitignore
│
└── templates/
    └── index.html
```

> ⚠️ `Mall_Customers.csv`, the training notebook, and any `.ipynb_checkpoints` are excluded via `.gitignore` — only `model.pkl`, `cluster_data.pkl` (a small 200×2 array needed for plotting), and the application code are deployed.

---

## 🚀 Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/mall-customer-segmentation.git
cd mall-customer-segmentation
```

### 2. Create Virtual Environment

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / Mac
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Trained Model & Cluster Data

Place these two files in the project root (generated from the training notebook — see below):

```text
model.pkl
cluster_data.pkl
```

### 5. Run Application

```bash
python app.py
```

### 6. Open Browser

```text
http://127.0.0.1:5000
```

---

## 🧠 Model Training Summary

1. **Feature selection** — `Annual Income (k$)` and `Spending Score (1-100)` (columns 3 and 4 of `Mall_Customers.csv`).
2. **Elbow Method** — WCSS computed for `k = 1` to `10` to determine the optimal number of clusters.
3. **Model** — `KMeans(n_clusters=5, init='k-means++', random_state=0)`.
4. **Output** — Each customer assigned a cluster label (`0`–`4`).

### ⚠️ Fixes applied to the original notebook before deployment

- `init='k-menas++'` (typo) corrected to `init='k-means++'` — scikit-learn validates this parameter strictly and raises `InvalidParameterError` on any misspelling.
- Added the Elbow Method plot cell (was computed but never visualized).
- Added a cell to **save the trained model** (`model.pkl`) — the original notebook never persisted it.
- Added a cell to save `cluster_data.pkl` (training `X` array, cluster labels `y`, and feature names) — required so the Flask app can render the comparison scatter plot without needing the original CSV.

```python
import pickle

pickle.dump(kmeans, open('model.pkl', 'wb'))

cluster_data = {
    'X': X,
    'y': y,
    'feature_names': ['Annual Income (k$)', 'Spending Score (1-100)']
}
pickle.dump(cluster_data, open('cluster_data.pkl', 'wb'))
```

---

## 💡 Usage

1. Enter the customer's **Annual Income (k$)**.
2. Enter the customer's **Spending Score (1–100)**.
3. Click **Find Segment**.
4. View the predicted cluster, a business-friendly label/description, and a scatter plot showing where this customer falls among existing segments.
5. Click **New Prediction** to check another customer, or **Go Back** to dismiss.

---

## 🎯 Prediction & Labeling Logic

```python
predicted_cluster = kmeans.predict([[annual_income, spending_score]])[0]
```

Segment labels are generated dynamically by comparing each cluster's centroid to the overall mean income/spending score — e.g. a centroid above both means is labeled **"High Income, High Spending"** and described as a premium customer segment. This avoids hardcoding fixed segment names, since K-Means cluster numbering is arbitrary and can change between training runs.

---

## ☁️ Deployment on Render

### Required Files

```text
requirements.txt
render.yaml
app.py
model.pkl
cluster_data.pkl
templates/index.html
```

### Deployment Steps

1. Push project to GitHub (dataset/notebook excluded via `.gitignore`).
2. Log in to https://render.com
3. Click **New +** → **Web Service**.
4. Connect your GitHub repository.
5. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn --workers 1 --timeout 60 app:app`
   - **Runtime:** Python
6. Click **Create Web Service**.

### Auto-deploy

Every `git push` to `main` triggers an automatic redeploy on Render.

---

## 🔮 Future Enhancements

- Allow users to upload their own customer CSV for batch segmentation.
- Add Age as a third clustering dimension with 3D visualization.
- Silhouette score display alongside the Elbow Method.
- Downloadable PDF/CSV report of segment assignment.
- Compare multiple `k` values interactively via a slider.

---

## 👥 Who Is This For?

- 🎓 Students learning unsupervised learning and clustering.
- 🤖 Machine Learning Engineers.
- 📊 Retail / marketing analytics enthusiasts.
- 🌐 Flask Developers.

---

## ⚠️ Disclaimer

> This project is intended for educational and learning purposes only.
>
> Segment assignments are based on a K-Means model trained on a small (200-row) sample dataset and should not be used for real business decisions without further validation on your own data.

---

## 👨‍💻 Author

**Suraj Prakash Verma**

- 🏢 UST Global
- 🌐 GitHub: https://github.com/surajprakashverma

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🌟 Show Your Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🛠️ Contribute improvements
📢 Share with fellow developers

Happy Coding! 🚀
