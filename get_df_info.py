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
    column_type: str = Field(description="Classification: 'numerical_continuous', 'numerical_discrete', 'categorical_low', 'categorical_high', 'high_cardinality', 'datetime', 'boolean', or 'identifier'")

class DataFrameColumnsInfo(BaseModel):
    columns: list[ColumnInfo] = Field(description="A list of objects, where each object contains the column name, description, data type, and column type classification")

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
    
    # Calculate unique counts for classification
    unique_counts = df.nunique().to_dict()
    
    prompt = f"""
    You are a helpful assistant that will give short descriptions for each column in the dataframe.
    {user_desc_part}
    The dataframe has {len(df)} rows and the following columns in EXACT order: {df.columns.tolist()}
    Dataframe info (data types, non-null counts, memory usage): {df.info()}
    Unique value counts for each column: {unique_counts}
    
    First 3 rows of data:
    {df.head(3).to_string()}
    
    Example values for each column:
    {example_values}
    
    IMPORTANT: You must return information for ALL columns in the EXACT same order as provided above.
    For each column, provide:
    1. The column name (must match exactly)
    2. A short description of what the column represents
    3. The data type of the column
    4. Column type classification based on these guidelines:
       - 'numerical_continuous': float/int with many unique values (like sales figures, prices, measurements)
       - 'numerical_discrete': int with limited unique values (like counts, years, ratings, scores)
       - 'categorical_low': object with few distinct categories (like gender, simple classifications)
       - 'categorical_high': object with many distinct categories but not too many (like companies, countries, departments)
       - 'high_cardinality': object with very many unique values (like names, IDs, text fields)
       - 'datetime': date/time columns
       - 'boolean': true/false columns
       - 'identifier': unique IDs or rankings
    
    Use your judgment to classify based on the nature of the data and cardinality ratios, not just the number of unique values.
    Consider the business context and how the column would be used in analysis.
    
    Please provide this information as a list of objects, where each object contains the column name, description, data type, and column type.
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
            'data_type': str(exact_dtypes.get(exact_col_name, col_info.data_type)),
            'column_type': col_info.column_type
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
            # Simple classification logic for fallback
            unique_count = df[col_name].nunique()
            dtype = str(exact_dtypes[col_name])
            
            if 'int' in dtype or 'float' in dtype:
                if unique_count > 50:
                    col_type = 'numerical_continuous'
                else:
                    col_type = 'numerical_discrete'
            else:
                if unique_count < 10:
                    col_type = 'categorical_low'
                elif unique_count < 100:
                    col_type = 'categorical_high'
                else:
                    col_type = 'high_cardinality'
            
            col_info = {
                'column_name': col_name,
                'description': f"Column: {col_name}",
                'data_type': dtype,
                'column_type': col_type
            }
            fallback_columns.append(col_info)
        
        return fallback_columns

result = llm_columns_info(df, user_desc)
print(f"Found {len(result)} columns:")
for col in result:
    print(f"  - {col['column_name']}: {col['description']} ({col['data_type']}, {col['column_type']})")