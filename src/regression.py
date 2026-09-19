import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD DATA
file_path = "F:/Codes/Projects/AI_Data_Analyst/data/HR_Employee_Attrition.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# SELECT FEATURES AND TARGET
features = [
    "Age",
    "JobLevel",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "Education",
    "Department",
    "JobRole"
]

# Target variable
target = "MonthlyIncome"

X = df[features]
y = df[target]

print("\nFeatures used:")
print(features)

print("\nTarget:")
print(target)

# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)

# NUMERICAL PREPROCESSING
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# CATEGORICAL PREPROCESSING
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# COMBINE PREPROCESSING
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# CREATE MACHINE LEARNING PIPELINE
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)

# TRAIN MODEL
model.fit(X_train, y_train)

print("\n================================")
print("Model trained successfully!")
print("================================")

# MAKE TEST PREDICTIONS
y_pred = model.predict(X_test)

# MODEL EVALUATION
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-------------------------")
print(f"MAE  : {mae:,.2f}")
print(f"RMSE : {rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# PREDICTION FUNCTION
def predict_salary(
    age,
    job_level,
    total_working_years,
    years_at_company,
    years_in_current_role,
    education,
    department,
    job_role
):
    """
    Predict an employee's MonthlyIncome.

    Parameters:
        age: Employee age
        job_level: Job level (1-5)
        total_working_years: Total years of work experience
        years_at_company: Years spent at the company
        years_in_current_role: Years in current role
        education: Education level (1-5)
        department: Department name
        job_role: Job role name

    Returns:
        Predicted MonthlyIncome
    """

    # Create a DataFrame containing the new employee
    employee = pd.DataFrame([
        {
            "Age": age,
            "JobLevel": job_level,
            "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "Education": education,
            "Department": department,
            "JobRole": job_role
        }
    ])

    # Make prediction
    prediction = model.predict(employee)

    # Return the first prediction
    return prediction[0]

'''
# TEST THE PREDICTION FUNCTION
predicted_salary = predict_salary(
    age=30,
    job_level=2,
    total_working_years=6,
    years_at_company=4,
    years_in_current_role=2,
    education=4,
    department="Research & Development",
    job_role="Research Scientist"
)

print("\n================================")
print("Salary Prediction")
print("================================")

print(f"Predicted Monthly Income: {predicted_salary:,.2f}")
'''