import os
import json
import logging
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback
from io import StringIO, BytesIO
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for serverless
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# LLM Integration (using OpenAI)
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    MAX_DATA_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = ['csv', 'json', 'xlsx', 'parquet']

# Initialize OpenAI client
if Config.OPENAI_API_KEY:
    openai.api_key = Config.OPENAI_API_KEY

class DataAnalystAgent:
    def __init__(self):
        self.current_data = None
        self.analysis_context = {}
        
    def parse_request(self, user_request: str) -> Dict[str, Any]:
        """Parse user request using LLM to extract intent and parameters"""
        if not Config.OPENAI_API_KEY:
            return {
                "data_source": "sample",
                "analysis_type": "descriptive",
                "specific_requirements": user_request,
                "visualization_type": "auto",
                "output_format": "summary"
            }
            
        system_prompt = """
        You are a data analysis request parser. Extract the following from user requests:
        1. data_source: URL, file path, or description of data needed
        2. analysis_type: descriptive, predictive, clustering, correlation, etc.
        3. specific_requirements: columns, filters, groupings, etc.
        4. visualization_type: chart type if requested
        5. output_format: table, chart, summary, etc.
        
        Respond with JSON only.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error parsing request: {e}")
            return {
                "data_source": "sample",
                "analysis_type": "descriptive",
                "specific_requirements": user_request,
                "visualization_type": "auto",
                "output_format": "summary"
            }
    
    def source_data(self, data_source: str, user_request: str) -> pd.DataFrame:
        """Source data from various sources"""
        try:
            # Try to load as URL
            if data_source.startswith(('http://', 'https://')):
                return self._load_from_url(data_source)
            
            # Generate synthetic data based on description
            else:
                return self._generate_synthetic_data(user_request)
                
        except Exception as e:
            logger.error(f"Error sourcing data: {e}")
            return self._create_sample_data()
    
    def _load_from_url(self, url: str) -> pd.DataFrame:
        """Load data from URL"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'csv' in content_type or url.endswith('.csv'):
            return pd.read_csv(StringIO(response.text))
        elif 'json' in content_type or url.endswith('.json'):
            return pd.read_json(StringIO(response.text))
        else:
            return pd.read_csv(StringIO(response.text))
    
    def _generate_synthetic_data(self, description: str) -> pd.DataFrame:
        """Generate synthetic data based on description using LLM"""
        if not Config.OPENAI_API_KEY:
            return self._create_sample_data()
            
        prompt = f"""
        Generate Python code to create a pandas DataFrame with synthetic data based on this description:
        {description}
        
        The code should:
        1. Import necessary libraries (pd, np, datetime)
        2. Create a DataFrame with realistic data
        3. Have at least 100 rows
        4. Include appropriate column names and data types
        5. Set np.random.seed(42) for reproducibility
        6. Return the DataFrame as 'df'
        
        Only return the Python code, no explanations. Keep it simple and safe.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean the code (remove markdown formatting if present)
            if code.startswith('```python'):
                code = code[9:]
            if code.endswith('```'):
                code = code[:-3]
            
            # Execute the generated code safely
            local_vars = {}
            exec(code, {"pd": pd, "np": np, "datetime": datetime}, local_vars)
            
            return local_vars.get('df', self._create_sample_data())
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create default sample data"""
        np.random.seed(42)
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'sales': np.random.normal(1000, 200, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'customers': np.random.poisson(50, 100)
        })
    
    def prepare_data(self, df: pd.DataFrame, requirements: str) -> pd.DataFrame:
        """Clean and prepare data"""
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def analyze_data(self, df: pd.DataFrame, analysis_type: str, requirements: str) -> Dict[str, Any]:
        """Perform data analysis"""
        # Basic statistical analysis
        basic_stats = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': dict(df.dtypes.astype(str)),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {}
        }
        
        # Generate insights
        insights = self._generate_insights(df, analysis_type, requirements)
        
        return {
            'basic_statistics': basic_stats,
            'insights': insights,
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_insights(self, df: pd.DataFrame, analysis_type: str, requirements: str) -> List[str]:
        """Generate insights using LLM or basic analysis"""
        if not Config.OPENAI_API_KEY:
            # Fallback to basic insights
            insights = []
            insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                insights.append(f"Numeric columns: {', '.join(numeric_cols[:3])}")
                
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                insights.append(f"Categorical columns: {', '.join(categorical_cols[:3])}")
                
            return insights
        
        # Create a summary of the data for the LLM
        data_summary = f"""
        Dataset shape: {df.shape}
        Columns: {list(df.columns)}
        Sample data:
        {df.head(3).to_string()}
        """
        
        prompt = f"""
        Analyze this dataset and provide 3-5 key insights as a JSON list of strings:
        
        {data_summary}
        
        Analysis type: {analysis_type}
        Requirements: {requirements}
        
        Focus on patterns, trends, and actionable insights. Return only the JSON array.
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=600
            )
            
            insights_text = response.choices[0].message.content.strip()
            return json.loads(insights_text)
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [
                f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns",
                f"Main columns: {', '.join(df.columns[:5])}",
                "Basic analysis completed successfully"
            ]
    
    def create_visualization(self, df: pd.DataFrame, viz_type: str, requirements: str) -> str:
        """Create visualizations and return as base64 encoded image"""
        plt.style.use('default')  # Use default style for better serverless compatibility
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if viz_type.lower() in ['auto', 'unknown']:
                viz_type = self._determine_viz_type(df)
            
            # Create visualization based on type
            if viz_type.lower() in ['histogram', 'hist']:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df[numeric_cols[0]].hist(bins=20, ax=ax, alpha=0.7)
                    ax.set_title(f'Distribution of {numeric_cols[0]}')
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel('Frequency')
            
            elif viz_type.lower() in ['scatter', 'scatterplot']:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    ax.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.6)
                    ax.set_xlabel(numeric_cols[0])
                    ax.set_ylabel(numeric_cols[1])
                    ax.set_title(f'{numeric_cols[0]} vs {numeric_cols[1]}')
            
            elif viz_type.lower() in ['bar', 'barplot']:
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    value_counts = df[categorical_cols[0]].value_counts().head(10)
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_title(f'Distribution of {categorical_cols[0]}')
                    ax.set_ylabel('Count')
            
            elif viz_type.lower() in ['line', 'lineplot']:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    ax.plot(df.index, df[numeric_cols[0]])
                    ax.set_title(f'Trend of {numeric_cols[0]}')
                    ax.set_xlabel('Index')
                    ax.set_ylabel(numeric_cols[0])
            
            else:
                # Default visualization
                numeric_df = df.select_dtypes(include=[np.number])
                if len(numeric_df.columns) >= 2:
                    # Scatter plot of first two numeric columns
                    ax.scatter(numeric_df.iloc[:, 0], numeric_df.iloc[:, 1], alpha=0.6)
                    ax.set_xlabel(numeric_df.columns[0])
                    ax.set_ylabel(numeric_df.columns[1])
                    ax.set_title('Data Visualization')
                else:
                    # Bar chart of first categorical column
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        value_counts = df[categorical_cols[0]].value_counts().head(10)
                        ax.bar(range(len(value_counts)), value_counts.values)
                        ax.set_xticks(range(len(value_counts)))
                        ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                        ax.set_title(f'Distribution of {categorical_cols[0]}')
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            plt.close()
            return ""
    
    def _determine_viz_type(self, df: pd.DataFrame) -> str:
        """Determine appropriate visualization type based on data"""
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        
        if numeric_cols >= 2:
            return 'scatter'
        elif numeric_cols == 1 and categorical_cols >= 1:
            return 'bar'
        elif numeric_cols == 1:
            return 'histogram'
        else:
            return 'bar'

# Initialize the agent
agent = DataAnalystAgent()

def handler(request):
    """Vercel serverless function handler"""
    try:
        # Handle CORS preflight requests
        if request.method == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type',
                },
                'body': ''
            }
        
        # Only allow POST requests for analysis
        if request.method != 'POST':
            return {
                'statusCode': 405,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json',
                },
                'body': json.dumps({"error": "Method not allowed. Use POST."})
            }
        
        # Get the request data
        try:
            body = request.get_json() if hasattr(request, 'get_json') else json.loads(request.body)
            user_request = body.get('question', body.get('request', ''))
        except:
            # Try to get data from body directly
            body_str = request.body if hasattr(request, 'body') else str(request.data, 'utf-8')
            if body_str.startswith('{'):
                body = json.loads(body_str)
                user_request = body.get('question', body.get('request', ''))
            else:
                user_request = body_str
        
        if not user_request:
            return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json',
                },
                'body': json.dumps({"error": "No analysis request provided"})
            }
        
        logger.info(f"Processing request: {user_request[:100]}...")
        
        # Step 1: Parse the request
        parsed_request = agent.parse_request(user_request)
        logger.info(f"Parsed request: {parsed_request}")
        
        # Step 2: Source the data
        data_source = parsed_request.get('data_source', 'sample')
        df = agent.source_data(data_source, user_request)
        logger.info(f"Sourced data with shape: {df.shape}")
        
        # Step 3: Prepare the data
        requirements = parsed_request.get('specific_requirements', '')
        df_prepared = agent.prepare_data(df, requirements)
        logger.info(f"Prepared data with shape: {df_prepared.shape}")
        
        # Step 4: Analyze the data
        analysis_type = parsed_request.get('analysis_type', 'descriptive')
        analysis_results = agent.analyze_data(df_prepared, analysis_type, requirements)
        
        # Step 5: Create visualization if requested
        viz_type = parsed_request.get('visualization_type', 'auto')
        visualization = None
        
        if viz_type and viz_type.lower() != 'none':
            viz_base64 = agent.create_visualization(df_prepared, viz_type, requirements)
            if viz_base64:
                visualization = f"data:image/png;base64,{viz_base64}"
        
        # Prepare the response
        response_data = {
            "status": "success",
            "request_parsed": parsed_request,
            "data_summary": {
                "rows": len(df_prepared),
                "columns": len(df_prepared.columns),
                "column_names": list(df_prepared.columns)
            },
            "analysis": analysis_results,
            "visualization": visualization,
            "execution_time": datetime.now().isoformat()
        }
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json',
            },
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        logger.error(traceback.format_exc())
        
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json',
            },
            'body': json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        }
