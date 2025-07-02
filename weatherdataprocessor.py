import pandas as pd # For structured data manipulation
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # Included for potential plotting
from datetime import datetime, timedelta # Handles timestamps and date arithmetic.
import sqlite3 # Connects to the SQLite database
from typing import Dict, List, Tuple # Adds type hints for better readability and error checking

# SECTION 1

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
            'temperature': (df['temperature'] < -100) | (df['temperature'] > 150),
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
            'temperature': (cleaned_df['temperature'] < -100) | (cleaned_df['temperature'] > 150),
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




# ------------------------------------------------------------------------------------------------

# SECTION 2


# Creating Derived Weather Features:
"""
Raw weather measurements tell only part of the story. Let's create derived 
features that provide more meaningful information for analysis and user experience:
"""

def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw weather data.
    """
    enhanced_df = df.copy()
    
    # Calculate Heat Index (how hot it feels with humidity)
    def calculate_heat_index(temp_f, humidity):
        """Calculate heat index in Fahrenheit."""
        if temp_f < 80:
            return temp_f
        
        hi = (0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094)))
        
        if hi > 79:
            hi = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
                  - 0.22475541 * temp_f * humidity - 0.00683783 * temp_f**2
                  - 0.05481717 * humidity**2 + 0.00122874 * temp_f**2 * humidity
                  + 0.00085282 * temp_f * humidity**2 - 0.00000199 * temp_f**2 * humidity**2)
        
        return hi
    
    # Convert temperature to Fahrenheit for heat index calculation
    temp_f = enhanced_df['temperature'] * 9/5 + 32
    enhanced_df['heat_index_f'] = temp_f.combine(enhanced_df['humidity'], calculate_heat_index)
    enhanced_df['heat_index_c'] = (enhanced_df['heat_index_f'] - 32) * 5/9
    
    # Calculate Wind Chill (how cold it feels with wind)
    def calculate_wind_chill(temp_f, wind_mph):
        """Calculate wind chill in Fahrenheit."""
        if temp_f > 50 or wind_mph < 3:
            return temp_f
        
        return (35.74 + 0.6215 * temp_f - 35.75 * (wind_mph ** 0.16) 
                + 0.4275 * temp_f * (wind_mph ** 0.16))
    
    # Convert wind speed from m/s to mph for wind chill calculation
    wind_mph = enhanced_df['wind_speed'] * 2.237
    enhanced_df['wind_chill_f'] = temp_f.combine(wind_mph, calculate_wind_chill)
    enhanced_df['wind_chill_c'] = (enhanced_df['wind_chill_f'] - 32) * 5/9
    
    # Comfort Index (combining temperature, humidity, and wind)
    def calculate_comfort_index(temp, humidity, wind_speed):
        """Calculate a comfort index from 0 (very uncomfortable) to 100 (very comfortable)."""
        # Ideal conditions: 20-25°C, 40-60% humidity, light breeze
        temp_comfort = 100 - abs(temp - 22.5) * 4  # Peak at 22.5°C
        humidity_comfort = 100 - abs(humidity - 50) * 2  # Peak at 50%
        wind_comfort = 100 - abs(wind_speed - 2) * 10  # Peak at 2 m/s
        
        # Combine with weights
        comfort = (temp_comfort * 0.5 + humidity_comfort * 0.3 + wind_comfort * 0.2)
        return max(0, min(100, comfort))
    
    enhanced_df['comfort_index'] = enhanced_df.apply(
        lambda row: calculate_comfort_index(row['temperature'], row['humidity'], row['wind_speed']), 
        axis=1
    )
    
    # Weather severity index
    def calculate_weather_severity(temp, wind_speed, humidity, pressure):
        """Calculate weather severity based on extreme conditions."""
        severity = 0
        
        # Temperature extremes
        if temp < -10 or temp > 40:
            severity += abs(temp - 15) * 0.5
        
        # High wind speeds
        if wind_speed > 10:
            severity += (wind_speed - 10) * 2
        
        # Very low or high humidity
        if humidity < 20 or humidity > 80:
            severity += abs(humidity - 50) * 0.3
        
        # Pressure extremes (indicating storms)
        if pressure < 1000 or pressure > 1030:
            severity += abs(pressure - 1015) * 0.2
        
        return min(100, severity)
    
    enhanced_df['weather_severity'] = enhanced_df.apply(
        lambda row: calculate_weather_severity(
            row['temperature'], row['wind_speed'], 
            row['humidity'], row['pressure']
        ), axis=1
    )
    
    # Time-based features
    enhanced_df['hour'] = enhanced_df.index.hour
    enhanced_df['day_of_week'] = enhanced_df.index.dayofweek
    enhanced_df['month'] = enhanced_df.index.month
    enhanced_df['is_weekend'] = enhanced_df['day_of_week'].isin([5, 6])
    
    # Temperature categories
    enhanced_df['temp_category'] = pd.cut(
        enhanced_df['temperature'],
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
        labels=['Freezing', 'Cold', 'Cool', 'Warm', 'Hot']
    )
    
    # Humidity categories
    enhanced_df['humidity_category'] = pd.cut(
        enhanced_df['humidity'],
        bins=[0, 30, 60, 80, 100],
        labels=['Dry', 'Comfortable', 'Humid', 'Very Humid']
    )
    
    # Seasonal indicators
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    enhanced_df['season'] = enhanced_df['month'].apply(get_season)
    
    return enhanced_df

def create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregated features for rolling windows and time periods.
    """
    agg_df = df.copy()
    
    # Rolling averages (last 24 hours)
    rolling_24h = df.rolling('24H')
    agg_df['temp_24h_avg'] = rolling_24h['temperature'].mean()
    agg_df['temp_24h_max'] = rolling_24h['temperature'].max()
    agg_df['temp_24h_min'] = rolling_24h['temperature'].min()
    agg_df['temp_24h_std'] = rolling_24h['temperature'].std()
    
    # Temperature change features
    agg_df['temp_change_1h'] = df['temperature'].diff(periods=1)
    agg_df['temp_change_6h'] = df['temperature'].diff(periods=6)
    agg_df['temp_change_24h'] = df['temperature'].diff(periods=24)
    
    # Pressure trends (indicating weather changes)
    agg_df['pressure_change_3h'] = df['pressure'].diff(periods=3)
    agg_df['pressure_trend'] = agg_df['pressure_change_3h'].apply(
        lambda x: 'Rising' if x > 1 else 'Falling' if x < -1 else 'Stable'
    )
    
    # Daily aggregations
    daily_agg = df.resample('D').agg({
        'temperature': ['mean', 'max', 'min', 'std'],
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'max'
    }).round(2)
    
    # Flatten column names
    daily_agg.columns = ['_'.join(col).strip() for col in daily_agg.columns]
    
    return agg_df, daily_agg








def prepare_analysis_datasets(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Prepare different dataset views for various types of analysis.
    """
    datasets = {}
    
    # Time series analysis dataset
    datasets['timeseries'] = df.resample('H').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean',
        'comfort_index': 'mean'
    }).round(2)
    
    # Daily summary dataset
    datasets['daily'] = df.resample('D').agg({
        'temperature': ['mean', 'max', 'min'],
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'max',
        'comfort_index': 'mean',
        'weather_severity': 'max'
    }).round(2)
    
    # City comparison dataset
    datasets['city_comparison'] = df.groupby(['city', 'country']).agg({
        'temperature': ['mean', 'std', 'max', 'min'],
        'humidity': 'mean',
        'comfort_index': 'mean',
        'weather_severity': 'mean'
    }).round(2)
    
    # Weather condition analysis
    datasets['weather_conditions'] = df.groupby('weather_main').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'pressure': 'mean',
        'wind_speed': 'mean',
        'comfort_index': 'mean'
    }).round(2)
    
    return datasets






# -----------------------------------------------------------------------------------------------------

# SECTION 3

from datetime import datetime, timedelta
import logging
from typing import Optional

class WeatherProcessingPipeline:
    """
    Automated pipeline for processing weather data from collection to analysis.
    """
    
    def __init__(self, database_path: str, processor: WeatherDataProcessor):
        self.database_path = database_path
        self.processor = processor
        self.logger = logging.getLogger(__name__)
        
    def process_recent_data(self, hours_back: int = 24) -> bool:
        """
        Process the most recent weather data.
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            self.logger.info(f"Processing data from {start_time} to {end_time}")
            
            # Load raw data
            raw_data = self.processor.load_data_from_db(
                start_date=start_time.strftime('%Y-%m-%d'),
                end_date=end_time.strftime('%Y-%m-%d')
            )
            
            if raw_data.empty:
                self.logger.warning("No data found for processing")
                return False
            
            # Quality assessment
            quality_report = self.processor.assess_data_quality(raw_data)
            self.logger.info(f"Quality assessment: {quality_report['total_records']} records")
            
            # Clean data
            cleaned_data = self.processor.clean_weather_data(raw_data)
            
            # Create derived features
            enhanced_data = self.processor.create_derived_features(cleaned_data)
            
            # Create aggregated features
            enhanced_data, daily_agg = self.processor.create_aggregated_features(enhanced_data)
            
            # Prepare analysis datasets
            analysis_datasets = self.processor.prepare_analysis_datasets(enhanced_data)
            
            # Store processed data
            self._store_processed_data(enhanced_data, daily_agg, analysis_datasets)
            
            self.logger.info("Data processing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return False
    
    def _store_processed_data(self, enhanced_data: pd.DataFrame, 
                            daily_agg: pd.DataFrame, 
                            analysis_datasets: Dict[str, pd.DataFrame]):
        """
        Store processed data back to the database.
        """
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Store enhanced data (replace existing processed data)
            enhanced_data.to_sql('processed_weather_data', conn, 
                               if_exists='replace', index=True)
            
            # Store daily aggregations
            daily_agg.to_sql('daily_weather_summary', conn, 
                           if_exists='replace', index=True)
            
            # Store analysis datasets
            for name, dataset in analysis_datasets.items():
                table_name = f'analysis_{name}'
                dataset.to_sql(table_name, conn, if_exists='replace', index=True)
            
            conn.commit()
            self.logger.info("Processed data stored successfully")
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to store processed data: {e}")
            raise
        finally:
            conn.close()
    
    def get_processing_summary(self) -> Dict:
        """
        Get a summary of the last processing run.
        """
        conn = sqlite3.connect(self.database_path)
        
        try:
            # Check when processed data was last updated
            cursor = conn.execute("""
                SELECT COUNT(*) as record_count, 
                       MIN(timestamp) as earliest_record,
                       MAX(timestamp) as latest_record
                FROM processed_weather_data
            """)
            result = cursor.fetchone()
            
            return {
                'processed_records': result[0],
                'earliest_record': result[1],
                'latest_record': result[2],
                'last_updated': datetime.now().isoformat()
            }
            
        except sqlite3.Error:
            return {'error': 'Could not retrieve processing summary'}
        finally:
            conn.close()

class DataQualityMonitor:
    """
    Monitor data quality over time and alert on issues.
    """
    
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.quality_thresholds = {
            'missing_data_percent': 10.0,  # Alert if >10% missing
            'impossible_values_percent': 1.0,  # Alert if >1% impossible
            'duplicate_percent': 0.5,  # Alert if >0.5% duplicates
            'data_gap_hours': 3.0  # Alert if gaps >3 hours
        }
    
    def check_quality_alerts(self, quality_report: Dict) -> List[Dict]:
        """
        Check if quality metrics exceed alert thresholds.
        """
        alerts = []
        
        # Check missing data
        total_missing = sum(info['percentage'] for info in quality_report['missing_values'].values())
        if total_missing > self.quality_thresholds['missing_data_percent']:
            alerts.append({
                'type': 'missing_data',
                'severity': 'warning',
                'message': f"High missing data rate: {total_missing:.1f}%"
            })
        
        # Check impossible values
        total_impossible = sum(quality_report['impossible_values'].values())
        impossible_percent = (total_impossible / quality_report['total_records']) * 100
        if impossible_percent > self.quality_thresholds['impossible_values_percent']:
            alerts.append({
                'type': 'impossible_values',
                'severity': 'error',
                'message': f"High impossible values rate: {impossible_percent:.1f}%"
            })
        
        # Check data gaps
        large_gaps = [gap for gap in quality_report['data_gaps'] 
                     if gap['duration'].total_seconds() / 3600 > self.quality_thresholds['data_gap_hours']]
        if large_gaps:
            alerts.append({
                'type': 'data_gaps',
                'severity': 'warning',
                'message': f"Found {len(large_gaps)} large data gaps"
            })
        
        return alerts

# Scheduling Automated Processing:
import schedule
import threading
import time

class ProcessingScheduler:
    """
    Schedule automated data processing tasks.
    """
    
    def __init__(self, pipeline: WeatherProcessingPipeline):
        self.pipeline = pipeline
        self.is_running = False
        self.scheduler_thread = None
        
    def start_scheduled_processing(self):
        """
        Start automated processing on a schedule.
        """
        # Process recent data every hour
        schedule.every().hour.do(self._process_recent_data)
        
        # Full quality check every 6 hours
        schedule.every(6).hours.do(self._full_quality_check)
        
        # Daily cleanup and optimization
        schedule.every().day.at("02:00").do(self._daily_maintenance)
        
        self.is_running = True
        
        def run_scheduler():
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        print("Processing scheduler started")
    
    def stop_scheduled_processing(self):
        """
        Stop the automated processing scheduler.
        """
        self.is_running = False
        schedule.clear()
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("Processing scheduler stopped")
    
    def _process_recent_data(self):
        """Process the last 2 hours of data."""
        self.pipeline.process_recent_data(hours_back=2)
    
    def _full_quality_check(self):
        """Perform a comprehensive quality check."""
        self.pipeline.process_recent_data(hours_back=24)
    
    def _daily_maintenance(self):
        """Perform daily database maintenance."""
        # This could include database optimization, old data cleanup, etc.
        print("Performing daily maintenance...")
