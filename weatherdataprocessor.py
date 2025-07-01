import pandas as pd # For structured data manipulation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # Included for potential plotting
from datetime import datetime, timedelta # Handles timestamps and date arithmetic.
import sqlite3 # Connects to the SQLite database
from typing import Dict, List, Tuple # Adds type hints for better readability and error checking

class WeatherDataProcessor:
    """
    Comprehensive weather data processing and quality assurance system.
    """
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.quality_report = {}
        # Stores the path to the SQLite database.
        # Initializes an empty dictionary to hold a data quality report.
        
    def load_data_from_db(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load weather data from database into a pandas DataFrame.
        """
        conn = sqlite3.connect(self.database_path) # Connects to SQLite database.
        
        query = """
        SELECT * FROM weather_readings 
        WHERE 1=1
        """
        # Builds a query with optional date filtering.

        params = []
        if start_date:
            query += " AND date(timestamp) >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date(timestamp) <= ?"
            params.append(end_date)
            
        query += " ORDER BY timestamp"
        
        # Executes the query
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
        # Outputs a clean DataFrame with time-series indexed weather data
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality assessment for weather data.
        """
        quality_report = {
            'total_records': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'missing_values': {},
            'impossible_values': {},
            'duplicates': 0,
            'outliers': {},
            'data_gaps': []
        }
        
        # Check for missing values - returns count and % missing per column
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][column] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Check for impossible values - flag potential errors
        impossible_conditions = {
            'temperature': (df['temperature'] < -100) | (df['temperature'] > 70),
            'humidity': (df['humidity'] < 0) | (df['humidity'] > 100),
            'pressure': (df['pressure'] < 800) | (df['pressure'] > 1200),
            'wind_speed': df['wind_speed'] < 0,
            'visibility': df['visibility'] < 0
        }
        
        for field, condition in impossible_conditions.items():
            if field in df.columns:
                impossible_count = condition.sum()
                if impossible_count > 0:
                    quality_report['impossible_values'][field] = impossible_count
        
        # Check for duplicates (same city, country, timestamp)
        duplicates = df.duplicated(subset=['city', 'country', df.index])
        quality_report['duplicates'] = duplicates.sum()
        
        # Check for outliers using IQR method
        numeric_columns = ['temperature', 'humidity', 'pressure', 'wind_speed']
        for column in numeric_columns:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))
                outlier_count = outlier_condition.sum()
                if outlier_count > 0:
                    quality_report['outliers'][column] = outlier_count
        
        # Check for data gaps (missing time periods)
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            expected_interval = time_diff.mode()[0] if len(time_diff.mode()) > 0 else pd.Timedelta(hours=1)
            
            gaps = time_diff[time_diff > expected_interval * 2]  # Gaps larger than 2x expected interval
            quality_report['data_gaps'] = [
                {'start': df.index[i-1], 'end': df.index[i], 'duration': gaps.iloc[i-1]}
                for i in gaps.index
            ]
        
        self.quality_report = quality_report
        return quality_report
    
    def print_quality_report(self, quality_report: Dict):
        """
        Print a human-readable data quality report.
        """
        print("=== WEATHER DATA QUALITY REPORT ===")
        print(f"Total Records: {quality_report['total_records']:,}")
        print(f"Date Range: {quality_report['date_range'][0]} to {quality_report['date_range'][1]}")
        print()
        
        # Missing values
        if quality_report['missing_values']:
            print("Missing Values:")
            for column, info in quality_report['missing_values'].items():
                print(f"  {column}: {info['count']} ({info['percentage']:.1f}%)")
        else:
            print("Missing Values: None detected")
        print()
        
        # Impossible values
        if quality_report['impossible_values']:
            print("Impossible Values:")
            for column, count in quality_report['impossible_values'].items():
                print(f"  {column}: {count} impossible values")
        else:
            print("Impossible Values: None detected")
        print()
        
        # Duplicates
        print(f"Duplicate Records: {quality_report['duplicates']}")
        print()
        
        # Outliers
        if quality_report['outliers']:
            print("Statistical Outliers:")
            for column, count in quality_report['outliers'].items():
                print(f"  {column}: {count} outliers")
        else:
            print("Statistical Outliers: None detected")
        print()
        
        # Data gaps
        if quality_report['data_gaps']:
            print("Data Gaps:")
            for gap in quality_report['data_gaps'][:5]:  # Show first 5 gaps
                print(f"  {gap['start']} to {gap['end']} (duration: {gap['duration']})")
            if len(quality_report['data_gaps']) > 5:
                print(f"  ... and {len(quality_report['data_gaps']) - 5} more gaps")
        else:
            print("Data Gaps: None detected")
    
    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean weather data by handling missing values, removing impossible values,
        and addressing other quality issues.
        Automatically fixes or mitigates data quality issues
        """
        cleaned_df = df.copy()
        cleaning_log = []
        
        # Remove exact duplicates
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates()
        if len(cleaned_df) < initial_count:
            cleaning_log.append(f"Removed {initial_count - len(cleaned_df)} duplicate records")
        
        # Handle impossible values by setting them to NaN (Not a Number)
        impossible_conditions = {
            'temperature': (cleaned_df['temperature'] < -100) | (cleaned_df['temperature'] > 70),
            'humidity': (cleaned_df['humidity'] < 0) | (cleaned_df['humidity'] > 100),
            'pressure': (cleaned_df['pressure'] < 800) | (cleaned_df['pressure'] > 1200),
            'wind_speed': cleaned_df['wind_speed'] < 0,
            'visibility': cleaned_df['visibility'] < 0
        }
        
        for field, condition in impossible_conditions.items():
            if field in cleaned_df.columns:
                impossible_count = condition.sum()
                if impossible_count > 0:
                    cleaned_df.loc[condition, field] = np.nan
                    cleaning_log.append(f"Set {impossible_count} impossible {field} values to NaN")
        
        # Handle missing values with appropriate strategies
        numeric_columns = ['temperature', 'feels_like', 'humidity', 'pressure', 'wind_speed', 'visibility']
        
        for column in numeric_columns:
            if column in cleaned_df.columns:
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    if missing_count / len(cleaned_df) < 0.05:  # Less than 5% missing
                        # Use linear interpolation for small gaps
                        # This means we assume a relation between the nearest values and create a data point between them
                        cleaned_df[column] = cleaned_df[column].interpolate(method='linear')
                        cleaning_log.append(f"Interpolated {missing_count} missing {column} values")
                    else:
                        # For larger gaps, use forward fill then backward fill
                        # This means carrying the closest values either forward or backward from the nearest data point
                        cleaned_df[column] = cleaned_df[column].fillna(method='ffill').fillna(method='bfill')
                        cleaning_log.append(f"Forward/backward filled {missing_count} missing {column} values")
        
        # Handle categorical missing values - If these values are missing, replace them with Unknown instead of NaN
        categorical_columns = ['weather_main', 'weather_description']
        for column in categorical_columns:
            if column in cleaned_df.columns:
                missing_count = cleaned_df[column].isnull().sum()
                if missing_count > 0:
                    cleaned_df[column] = cleaned_df[column].fillna('Unknown')
                    cleaning_log.append(f"Filled {missing_count} missing {column} values with 'Unknown'")
        
        # Log cleaning operations - Will print the cleaning operation to inform the user
        print("=== DATA CLEANING LOG ===")
        for log_entry in cleaning_log:
            print(f"✓ {log_entry}")
        
        return cleaned_df

# Visualizing Data Quality:
def create_quality_visualizations(self, df: pd.DataFrame):
    """
    Create visualizations to help assess data quality.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Weather Data Quality Assessment', fontsize=16)
    
    # Missing data heatmap
    missing_data = df.isnull()
    axes[0,0].imshow(missing_data.T, cmap='Reds', aspect='auto')
    axes[0,0].set_title('Missing Data Pattern')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Columns')
    
    # Temperature outliers
    temp_data = df['temperature'].dropna()
    Q1 = temp_data.quantile(0.25)
    Q3 = temp_data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = temp_data[(temp_data < (Q1 - 1.5 * IQR)) | (temp_data > (Q3 + 1.5 * IQR))]
    
    axes[0,1].boxplot(temp_data)
    axes[0,1].scatter(np.ones(len(outliers)), outliers, color='red', alpha=0.6)
    axes[0,1].set_title('Temperature Distribution with Outliers')
    axes[0,1].set_ylabel('Temperature (°C)')
    
    # Data completeness over time
    daily_completeness = df.resample('D').count()['temperature'] / df.resample('D').size().replace(0, np.nan)
    axes[1,0].plot(daily_completeness.index, daily_completeness * 100)
    axes[1,0].set_title('Daily Data Completeness')
    axes[1,0].set_ylabel('Completeness (%)')
    axes[1,0].set_xlabel('Date')
    
    # Correlation between weather variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    im = axes[1,1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    axes[1,1].set_title('Weather Variable Correlations')
    axes[1,1].set_xticks(range(len(numeric_cols)))
    axes[1,1].set_yticks(range(len(numeric_cols)))
    axes[1,1].set_xticklabels(numeric_cols, rotation=45)
    axes[1,1].set_yticklabels(numeric_cols)
    
    plt.tight_layout()
    plt.show()
