import pandas as pd
import sqlite3

# Load the CSV data into a DataFrame with the correct delimiter
df = pd.read_csv(r'C:\Users\franc\OneDrive\Documents\ITDAA-Project\heart.csv', delimiter=';')

# Create a SQLite database connection
conn = sqlite3.connect('heart_disease.db')

# Create the table in the SQLite database
conn.execute('''
CREATE TABLE IF NOT EXISTS heart_data (
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak REAL,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER
)
''')

# Insert the data into the SQLite database
df.to_sql('heart_data', conn, if_exists='replace', index=False)

# Commit and close the connection
conn.commit()
conn.close()

print("Data inserted into the SQL database successfully.")
