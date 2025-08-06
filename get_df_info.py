import pandas as pd
from pydantic import BaseModel, Field
from typing import Dict
from dotenv import load_dotenv
from llm_client import generate_structured

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

def build_prompt(df, user_desc, example_values):
    """Build the prompt for LLM column analysis"""
    user_desc_part = f"The user has provided the following description of the dataframe: {user_desc}\n" if user_desc.strip() else ""
    
    prompt = f"""
    You are a helpful assistant that will give short descriptions for each column in the dataframe.
    {user_desc_part}
    The dataframe has the following columns in EXACT order: {df.columns.tolist()}
    The dataframe has the following data types: {df.dtypes.to_dict()}
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
    return prompt

def validate_and_fix_columns(columns_info, exact_columns, exact_dtypes):
    """Validate and fix column names to match exact dataframe columns"""
    validated_columns = []
    
    for i, col_info in enumerate(columns_info.columns):
        # Ensure we use the exact column name from the dataframe
        exact_col_name = exact_columns[i] if i < len(exact_columns) else col_info.column_name
        
        validated_col = {
            'column_name': exact_col_name,
            'description': col_info.description,
            'data_type': str(exact_dtypes.get(exact_col_name, col_info.data_type))
        }
        validated_columns.append(validated_col)
    
    return validated_columns

def llm_columns_info(df, user_desc):
    """Get column information from LLM with validation"""
    # Get example values for each column
    example_values = get_example_values(df)
    
    # Store the exact column order and names
    exact_columns = df.columns.tolist()
    exact_dtypes = df.dtypes.to_dict()
    
    # Build the prompt
    prompt = build_prompt(df, user_desc, example_values)
    
    try:
        # Use the new LLM client for structured generation
        columns_info = generate_structured(
            prompt=prompt,
            response_schema=DataFrameColumnsInfo
        )
        
        # Validate and fix column names to match exact dataframe columns
        validated_info = validate_and_fix_columns(columns_info, exact_columns, exact_dtypes)
        
        return validated_info
        
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        # Fallback: create basic column info from dataframe
        fallback_columns = []
        for col_name in exact_columns:
            col_info = {
                'column_name': col_name,
                'description': f"Column: {col_name}",
                'data_type': str(exact_dtypes[col_name])
            }
            fallback_columns.append(col_info)
        
        return fallback_columns

result = llm_columns_info(df, user_desc)
print(f"Found {len(result)} columns:")
for col in result:
    print(f"  - {col['column_name']}: {col['description']} ({col['data_type']})")