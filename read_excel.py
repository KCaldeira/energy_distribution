import pandas as pd

# Read the Excel file
df = pd.read_excel('Energy Distribution Input 2025-03-04 - Test5.xlsx')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)

print("\nDataFrame Info:")
print("-" * 50)
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {len(df.columns)}")
print("\nColumn names:")
for col in df.columns:
    print(f"- {col}")

print("\nFirst 10 rows of data:")
print("-" * 50)
print(df.head(10).to_string()) 