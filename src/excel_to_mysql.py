import pandas as pd
from sqlalchemy import create_engine

# 1. Read Excel file
file_path = "F:/Codes/Projects/AI_Data_Analyst/data/HR_Employee_Attrition.xlsx"

df = pd.read_excel(file_path)

print("Excel file loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. MySQL connection
username = "root"
password = "1234"
host = "localhost"
database = "ai_employee_data"

engine = create_engine(
    f"mysql+pymysql://{username}:{password}@{host}/{database}"
)

# 3. Upload DataFrame to MySQL
df.to_sql(
    name="employees",
    con=engine,
    if_exists="replace",
    index=False
)

print("Data successfully imported into MySQL!")