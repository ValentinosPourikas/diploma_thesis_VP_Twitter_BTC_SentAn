'''
Bitcoin CSV Data Processor

This script processes historical data from a CSV file acquired from investing.com. The data is preprocessed and augmented to calculate mean values for each entry. The processed data is then saved as both a CSV file and a JSON dictionary.

The processing steps include:
1. Conversion of the date format to Y-M-D.
2. Calculation of mean values for open-close prices (mean_OC) and low-high values (mean_LH).
3. Conversion of numeric strings with commas and 'K'/'M' suffix to float.
4. Saving the processed data to a CSV file and a JSON file.

Input:
- 'BTC_USD_historical.csv': Input CSV file containing historical data with columns: date, close price, open price, high value, low value, volume, and change.
- Output file paths: 'BTC_values_augmented.csv' (processed CSV) and 'BTC_values_augmented.json' (processed JSON).

Output:
- 'BTC_values_augmented.csv': Processed CSV file with additional columns for mean_OC and mean_LH.
- 'BTC_values_augmented.json': Processed JSON file containing a dictionary with processed data entries, including mean values.
Note: The two output files 'BTC_values_augmented.csv' and 'BTC_values_augmented.json' contain the same data but in different formats. Only the '.json' file is intended for eventual use due to its compatibility with various applications and services.

'''
#%%
from datetime import datetime
import csv
import json
import os
import logging
import time 

DATE_COLUMN = 0
CLOSE_PRICE_COLUMN = 1
OPEN_PRICE_COLUMN = 2
HIGH_VALUE_COLUMN = 3
LOW_VALUE_COLUMN = 4
VOLUME_COLUMN = 5
CHANGE_COLUMN = 6

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Record the start time
start_time = time.time()

# Initialize error counter
error_count = 0

# Function to convert date string to the desired format
def convert_date(date_str):
    '''
    Convert date string to the desired format.
    Args:
    - date_str (str): Date string in the format "Month D, Y".
    Returns:
    - str: Date string in the format "Y-M-D".
    '''
    try:
        date_obj = datetime.strptime(date_str, "%b %d, %Y")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        logging.error(f"Error converting date: {date_str}")
    
        return None

# Function to convert numeric strings with commas and 'K'/'M' suffix to float. None values stay nonexistent
def convert_numeric_string(num_str):
    '''
    Convert numeric strings with commas and 'K'/'M' suffix to float.
    Args:
    - num_str (str): Numeric string to be converted.
    Returns:
    - float or str: Converted float value or original string.
    '''
    num_str = num_str.replace(',', '')
    if num_str.endswith('K'):
        num_str = num_str[:-1]
        return round(float(num_str) * 1000, 1)
    if num_str.endswith('M'):
        num_str = num_str[:-1]
        return round(float(num_str) * 1000000, 1)
    if num_str.endswith('%'):
        #logging.info(f"Encountered percentage value: {num_str}")
        return num_str
    if num_str == "-":
        #logging.info(f"Encountered '-' value")
        return num_str
    try:
        return float(num_str)
    except ValueError:
        logging.error(f"Error converting numeric string: {num_str}")
        global error_count
        error_count += 1
        return num_str
# Process CSV data to extract dates and calculate mean values
def process_csv(input_file, output_csv):
    '''
    Process CSV data to extract dates and calculate mean values.
    Args:
    - input_file (str): Path to the input CSV file.
    - output_csv (str): Path to the output CSV file.
    Returns:
    - tuple: A tuple containing header, dates, mean low-high values, mean open-close values, and original data.
    '''
    dates = []
    mean_LH_values = []
    mean_OC_values = []
    original_data = []

    try:
        with open(input_file, 'r') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)

            for row in csv_reader:
                original_data.append(row)  # Store original data
                date_str = row[0].strip('"')
                formatted_date = convert_date(date_str)
                dates.append(formatted_date)

                close_price = float(row[CLOSE_PRICE_COLUMN].replace(",", ""))
                open_price = float(row[OPEN_PRICE_COLUMN].replace(",", ""))
                high_value = float(row[HIGH_VALUE_COLUMN].replace(",", ""))
                low_value = float(row[LOW_VALUE_COLUMN].replace(",", ""))
                # volume = convert_numeric_string(row[VOLUME_COLUMN])
                # change = convert_numeric_string(row[CHANGE_COLUMN])

                mean_LH = round((low_value + high_value) / 2, 1)
                mean_OC = round((open_price + close_price) / 2, 1)
                mean_LH_values.append(mean_LH)
                mean_OC_values.append(mean_OC)
    except FileNotFoundError:
        logging.error("Input CSV file not found.")
        return None

    return header, dates, mean_LH_values, mean_OC_values, original_data

# Write processed data to a CSV file
def write_csv(output_csv, header, dates, mean_LH_values, mean_OC_values, original_data):
    '''
    Write processed data to a CSV file.

    Args:
    - output_csv (str): Path to the output CSV file.
    - header (list): List of column headers.
    - dates (list): List of processed dates.
    - mean_LH_values (list): List of mean low-high values.
    - mean_OC_values (list): List of mean open-close values.
    - original_data (list): List of original data rows.
    '''
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header + ['mean_OC', 'mean_LH'])
            for date, mean_LH, mean_OC, original_row in zip(dates, mean_LH_values, mean_OC_values, original_data):
                csv_writer.writerow([date] + list(map(convert_numeric_string, [original_row[CLOSE_PRICE_COLUMN], original_row[OPEN_PRICE_COLUMN], original_row[HIGH_VALUE_COLUMN], original_row[LOW_VALUE_COLUMN], original_row[VOLUME_COLUMN],original_row[CHANGE_COLUMN]])) + [mean_LH, mean_OC])
    except Exception as e:
        logging.error(f"Error writing CSV file: {e}")

# Write processed data to a JSON file
def write_json(output_json, header, dates, mean_LH_values, mean_OC_values, original_data):
    '''
    Write processed data to a JSON file.
    Args:
    - output_json (str): Path to the output JSON file.
    - header (list): List of column headers.
    - dates (list): List of processed dates.
    - mean_LH_values (list): List of mean low-high values.
    - mean_OC_values (list): List of mean open-close values.
    - original_data (list): List of original data rows.
    '''
    data_to_write = {}
    for date, mean_LH, mean_OC, original_row in zip(dates, mean_LH_values, mean_OC_values, original_data):
        data_row = {
            **{header[i + 1]: convert_numeric_string(value) for i, value in enumerate(original_row[1:7])},
            'mean_LowHigh': mean_LH,
            'mean_OpenClose': mean_OC,
        }
        data_to_write[date] = data_row

    try:
        with open(output_json, 'w') as jsonfile:
            json.dump(data_to_write, jsonfile, indent=4)
    except Exception as e:
        logging.error(f"Error writing JSON file: {e}")

# File paths
input_file = 'BTC_USD_historical.csv'
output_csv = 'BTC_values_augmented.csv'
output_json = 'BTC_values_augmented.json'

# Process and write data
header, dates, mean_LH_values, mean_OC_values, original_data = process_csv(input_file, output_csv)
if header is not None:
    write_csv(output_csv, header, dates, mean_LH_values, mean_OC_values, original_data)
    write_json(output_json, header, dates, mean_LH_values, mean_OC_values, original_data)

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the time taken to run the full program
logging.info(f"Elapsed time: {elapsed_time} seconds.")
logging.info(f"Total errors encountered: {error_count}")
