import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib

# Đọc dữ liệu
dataset_df = pd.read_csv(r'VN_housing_dataset.csv')

# Data clean
dataset_df=dataset_df.drop(52317, axis=0, inplace=False)
dataset_df=dataset_df.drop(55653, axis=0, inplace=False)
dataset_df=dataset_df.drop(58003, axis=0, inplace=False)
dataset_df=dataset_df.drop(63, axis=0, inplace=False)

dataset_df['Giá/m2'] = dataset_df['Giá/m2'].replace(to_replace=',', value='.', regex=True)
dataset_df['Giá/m2'] = dataset_df['Giá/m2'].replace('[^\d.]', '', regex=True).astype(float)

dataset_df['Dài'] = dataset_df['Dài'].replace('[^\d.]', '', regex=True)
dataset_df['Dài'] = dataset_df['Dài'].replace('[^\d.]', '', regex=True).astype(float)

dataset_df['Rộng'] = dataset_df['Rộng'].replace('[^\d.]', '', regex=True)
dataset_df['Rộng'] = dataset_df['Rộng'].replace('[^\d.]', '', regex=True).astype(float)

dataset_df['Số phòng ngủ'] = dataset_df['Số phòng ngủ'].replace('[^\d.]', '', regex=True)
dataset_df['Số phòng ngủ'] = dataset_df['Số phòng ngủ'].replace('[^\d.]', '', regex=True).astype(float)

dataset_df['Diện tích'] = dataset_df['Diện tích'].replace('[^\d.]', '', regex=True)
dataset_df['Diện tích'] = dataset_df['Diện tích'].replace('[^\d.]', '', regex=True).astype(float)



dataset_df = dataset_df.rename(columns={'Địa chỉ': 'Dia_chi'})
dataset_df = dataset_df.rename(columns={'Loại hình nhà ở': 'Loai_nha'})
dataset_df = dataset_df.rename(columns={'Giấy tờ pháp lý': 'Giay_to'})
dataset_df = dataset_df.rename(columns={'Số tầng': 'So_tang'})
dataset_df = dataset_df.rename(columns={'Số phòng ngủ': 'So_phong'})
dataset_df = dataset_df.rename(columns={'Diện tích': 'Dien_tich'})

dataset_df=dataset_df.dropna()

y = dataset_df['Giá/m2']
X = dataset_df.drop('Giá/m2', axis=1)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo các pipeline xử lý dữ liệu
numerical_cols = ['Dien_tich', 'Dài', 'Rộng']  # thay thế các giá trị thực tế của bạn
categorical_cols = ['Dia_chi', 'Quận', 'Huyện', 'Loai_nha', 'Giay_to']  # thay thế các giá trị thực tế của bạn

#
my_cols=categorical_cols+numerical_cols
X_train = X_train[my_cols]
X_test = X_test[my_cols]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Định nghĩa lại my_pipeline với một RandomForestRegressor mặc định
my_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))
])

# Function to create the model with hyperparameter space
# Define the parameter distribution
param_dist = {
    'model__n_estimators': randint(100, 1000),
    'model__max_depth': randint(5, 30),
    'model__min_samples_split': randint(2, 11)
}

# Create a RandomizedSearchCV object and fit it to the training data
random_search = RandomizedSearchCV(
    estimator=my_pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

# Print the best parameters
print("Best parameters found: ", random_search.best_params_)

# Your best model is now fitted with the best set of hyperparameters
best_model = random_search.best_estimator_

# Use the best model to make predictions
val_preds = best_model.predict(X_test)
print("Validation MAE: ", mean_absolute_error(y_test, val_preds))

# Save the best model using joblib
joblib.dump(best_model, 'best_model.pkl')