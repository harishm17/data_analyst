import pandas as pd
from google import genai
from pydantic import BaseModel, Field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()


df = pd.read_csv('data/vgsales.csv')
user_desc = ''

class ColumnInfo(BaseModel):
    column_name: str = Field(description="The exact name of the column")
    description: str = Field(description="A short description of what the column represents")
    data_type: str = Field(description="The data type of the column")

class DataFrameColumnsInfo(BaseModel):
    columns: list[ColumnInfo] = Field(description="A list of objects, where each object contains the column name, description, and data type")

def get_example_values(df, num_examples=5):
    """Extract example values for each column in the dataframe."""
    example_values = {}
    for col in df.columns:
        # Get non-null values and take first few unique ones
        non_null_values = df[col].dropna().unique()
        example_values[col] = non_null_values[:num_examples].tolist()
    return example_values

def llm_columns_info(df, user_desc):
    # Get example values for each column
    example_values = get_example_values(df)
    
    # Store the exact column order and names
    exact_columns = df.columns.tolist()
    exact_dtypes = df.dtypes.to_dict()
    
    # Build prompt conditionally based on user description
    user_desc_part = f"The user has provided the following description of the dataframe: {user_desc}\n" if user_desc.strip() else ""
    
    prompt = f"""
    You are a helpful assistant that will give short descriptions for each column in the dataframe.
    {user_desc_part}
    The dataframe has the following columns in EXACT order: {exact_columns}
    The dataframe has the following data types: {exact_dtypes}
    Describe function output: {df.describe().to_dict()}
    
    Example values for each column:
    {example_values}
    
    IMPORTANT: You must return information for ALL columns in the EXACT same order as provided above.
    For each column, provide:
    1. The column name (must match exactly)
    2. A short description of what the column represents
    3. The data type of the column
    
    Please provide this information as a list of objects, where each object contains the column name, description, and data type.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": DataFrameColumnsInfo,
        },
    )

    columns_info: DataFrameColumnsInfo = response.parsed

    # Verify and fix column names to match exact dataframe columns
    validated_columns = []
    
    for i, col_info in enumerate(columns_info.columns):
        # Ensure we use the exact column name from the dataframe
        exact_col_name = exact_columns[i] if i < len(exact_columns) else col_info.column_name
        
        validated_col = ColumnInfo(
            column_name=exact_col_name,
            description=col_info.description,
            data_type=str(exact_dtypes.get(exact_col_name, col_info.data_type))
        )
        validated_columns.append(validated_col)
    
    # Create a new DataFrameColumnsInfo with validated columns
    validated_info = DataFrameColumnsInfo(columns=validated_columns)
    
    print(example_values)
    return validated_info

llm_columns_info(df, user_desc)