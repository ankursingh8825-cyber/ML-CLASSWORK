import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("csv_files/student_academic_placement_performance_dataset.csv")
print("Dataset Loaded! Columns:", df.columns.tolist())

# ==================================
# ENCODE CATEGORICAL COLUMNS
# ==================================
cat_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# ==================================
# TARGET VARIABLES
# ==================================
y_reg = df['salary_package_lpa']        # Regression target
y_clf = df['placement_status']          # Classification target

# ==================================
# FEATURE SET (Remove targets & ID)
# ==================================
X = df.drop(['student_id', 'salary_package_lpa', 'placement_status'], axis=1)

# ==================================
# SPLIT DATA
# ==================================
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)

# ==================================
# 1. MULTIPLE LINEAR REGRESSION
# ==================================
lin_reg = LinearRegression()
lin_reg.fit(X_train_r, y_train_r)
pred_lin = lin_reg.predict(X_test_r)

print("\n--- Multiple Linear Regression ---")
print("R2 Score:", r2_score(y_test_r, pred_lin))
print("MSE:", mean_squared_error(y_test_r, pred_lin))

# ==================================
# 2. KNN REGRESSION
# ==================================
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train_r, y_train_r)
pred_knn_r = knn_reg.predict(X_test_r)

print("\n--- KNN Regression ---")
print("R2 Score:", r2_score(y_test_r, pred_knn_r))
print("MSE:", mean_squared_error(y_test_r, pred_knn_r))

# ==================================
# 3. LOGISTIC REGRESSION (Classification)
# ==================================
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_c, y_train_c)
pred_log = log_clf.predict(X_test_c)

print("\n--- Logistic Regression Classification ---")
print("Accuracy:", accuracy_score(y_test_c, pred_log))

# ==================================
# 4. NAIVE BAYES CLASSIFIER
# ==================================
nb_clf = GaussianNB()
nb_clf.fit(X_train_c, y_train_c)
pred_nb = nb_clf.predict(X_test_c)

print("\n--- Naive Bayes Classification ---")
print("Accuracy:", accuracy_score(y_test_c, pred_nb))

# ==================================
# 5. KNN CLASSIFIER
# ==================================
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train_c, y_train_c)
pred_knn_c = knn_clf.predict(X_test_c)

print("\n--- KNN Classification ---")
print("Accuracy:", accuracy_score(y_test_c, pred_knn_c))

# ==================================
# 6. DECISION TREE CLASSIFIER
# ==================================
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train_c, y_train_c)
pred_dt = dt_clf.predict(X_test_c)

print("\n--- Decision Tree Classification ---")
print("Accuracy:", accuracy_score(y_test_c, pred_dt))

