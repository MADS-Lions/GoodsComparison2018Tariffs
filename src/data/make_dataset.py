# -*- coding: utf-8 -*-
"""This script loads the raw data, manipulates it, and saves the processed data. This script also contains functions to check for missing values, data types, and unit tests for the data.
"""
#import relevant libraries
import click # type: ignore
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv # type: ignore
import pandas as pd # type: ignore


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
        Args:
            input_filepath::str: The path to the raw data file.
            output_filepath::str: The path to the processed data file.
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    Canadian_CPI_dataset = load_data("../../data/Canadian_CPI.csv")
    USA_CPI_dataset = load_data("../../data/USA_CPI.csv")
    Canadian_CPI_dataset = manipulate_data(Canadian_CPI_dataset, "Canadian")
    USA_CPI_dataset = manipulate_data(USA_CPI_dataset, "USA")
    save_data(Canadian_CPI_dataset, "../../data/processed/Canadian_CPI_Processed.csv")
    save_data(USA_CPI_dataset, "../../data/processed/USA_CPI_Processed.csv")
    logger.info('Data processing completed successfully')

def load_data(input_filepath, which_data_file, header = 0):
    """Load data from input_filepath and return a pandas dataframe.
    Args:
        input_filepath::str: The path to the raw data file.
        which_data_file::str: The type of data file to load.
    Returns:
        data::pandas_df: The loaded data.
    """
    if which_data_file == '.csv':
        data = pd.read_csv(input_filepath, header = header)
    else:
        data = pd.read_excel(input_filepath, header = header)
    return data

def manipulate_data(data, which_data_file):
    """Load data and manipulate to return a pandas dataframe ready for visualization and analysis.
    Args:
        data::pandas_df: The loaded data.
        which_data_file::str: The data file to load (Canada or US).
        Returns:
        data::pandas_df: The manipulated and formatted data.
    """
    if which_data_file == "Canada":
        if 'Products and product groups' in data.columns:
            
            Canadian_CPI_dataset = data.copy()
            Canadian_CPI_dataset['REF_DATE'] = pd.to_datetime(Canadian_CPI_dataset['REF_DATE'])
            Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['REF_DATE']>='2017-01-01')&(Canadian_CPI_dataset['REF_DATE']<='2020-02-01')]
            print('UOM has two unique values in the dataset: ', Canadian_CPI_dataset['UOM'].unique())
            Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['UOM']=='2002=100')]
            print('UOM now only has 2002 value: ', Canadian_CPI_dataset['UOM'].unique())
            Canadian_CPI_dataset = Canadian_CPI_dataset[['Products and product groups', 'REF_DATE', 'VALUE']]
            Canadian_CPI_dataset = Canadian_CPI_dataset.groupby(['Products and product groups', 'REF_DATE']).mean()    
            Canadian_CPI_dataset.reset_index(inplace=True)
            Canadian_CPI_dataset['REF_DATE'] = pd.to_datetime(Canadian_CPI_dataset['REF_DATE'])
            return Canadian_CPI_dataset
        else: 
            Canadian_CPI_dataset = data.copy()
            Canadian_CPI_dataset['REF_DATE'] = pd.to_datetime(Canadian_CPI_dataset['REF_DATE'])
            Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['REF_DATE']>='2017-01-01')&(Canadian_CPI_dataset['REF_DATE']<='2020-02-01')]
            print('UOM has two unique values in the dataset: ', Canadian_CPI_dataset['UOM'].unique())
            try:
                Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['UOM']=='Dollars')&(Canadian_CPI_dataset['Seasonal adjustment']=='Seasonally adjusted')]
            except:
                Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['UOM']=='Dollars')&(Canadian_CPI_dataset['Adjustments']=='Seasonally adjusted')]
            print('UOM now only has 2002 value: ', Canadian_CPI_dataset['UOM'].unique())
            try: 
                Canadian_CPI_dataset = Canadian_CPI_dataset[[
                    'Principal statistics',
                    'North American Industry Classification System (NAICS)',
                    'REF_DATE',
                    'VALUE'
                ]]
            except:
                Canadian_CPI_dataset = Canadian_CPI_dataset[[
                    'Sales',
                    'North American Industry Classification System (NAICS)',
                    'REF_DATE',
                    'VALUE'
                ]]
            Canadian_CPI_dataset['North American Industry Classification System (NAICS)'] = (
                Canadian_CPI_dataset['North American Industry Classification System (NAICS)'].str.replace(r'[\s*]\[[\d*]\]', '', regex=True)
            )

            
            return Canadian_CPI_dataset
    else:
        USA_CPI_dataset = data.copy()
        USA_CPI_dataset = USA_CPI_dataset[USA_CPI_dataset['DATA_TYPE'] == 'SEASONALLY ADJUSTED INDEX']
        USA_CPI_dataset = USA_CPI_dataset[(USA_CPI_dataset['YEAR']>=2017)&(USA_CPI_dataset['YEAR']<=2020)]
        USA_CPI_dataset.drop(columns = ['ITEM', 'seriesid', 'DATA_TYPE'], inplace=True)
        USA_CPI_dataset = USA_CPI_dataset.groupby(['TITLE', 'YEAR']).mean()
        USA_CPI_dataset_T = USA_CPI_dataset.T
        df_USA_Single_Columns_for_2017_2018_2019_2020 = pd.DataFrame()
        
        for col in USA_CPI_dataset_T.columns:
            
            col_name = col[0]
            year = col[1]
            if year == 2017:
                add_column_2017 = USA_CPI_dataset_T[col].tolist()
                new_index = [str(ind) + '_2017' for ind in USA_CPI_dataset_T.index]
                dataframe_2017 = pd.DataFrame(add_column_2017, columns = [col_name], index = new_index)
            elif year == 2018:
                add_column_2018 = USA_CPI_dataset_T[col].tolist()
                new_index = [str(ind) + '_2018' for ind in USA_CPI_dataset_T.index]
                dataframe_2018 = pd.DataFrame(add_column_2018, columns = [col_name], index = new_index)
            elif year == 2019:
                add_column_2019 = USA_CPI_dataset_T[col].tolist()
                new_index = [str(ind) + '_2019' for ind in USA_CPI_dataset_T.index]
                dataframe_2019 = pd.DataFrame(add_column_2019, columns = [col_name], index = new_index)
            elif year == 2020:
                add_column_2020 = USA_CPI_dataset_T[col].tolist()
                new_index = [str(ind) + '_2020' for ind in USA_CPI_dataset_T.index]
                dataframe_2020 = pd.DataFrame(add_column_2020, columns = [col_name], index = new_index)
                column_to_add = pd.concat([dataframe_2017, dataframe_2018, dataframe_2019, dataframe_2020], axis=0)
                df_USA_Single_Columns_for_2017_2018_2019_2020 = pd.concat([df_USA_Single_Columns_for_2017_2018_2019_2020, column_to_add], axis=1)
        df_USA_Single_Columns_for_2017_2018_2019_2020.reset_index(inplace=True)
        df_USA_Single_Columns_for_2017_2018_2019_2020.rename(columns = {'index': 'Date'}, inplace=True)

        df_USA_Single_Columns_for_2017_2018_2019_2020['Month'] = df_USA_Single_Columns_for_2017_2018_2019_2020['Date'].str.split('_').str[0]

        df_USA_Single_Columns_for_2017_2018_2019_2020['Year'] = df_USA_Single_Columns_for_2017_2018_2019_2020['Date'].str.split('_').str[1]
        df_USA_Single_Columns_for_2017_2018_2019_2020['REF_DATE'] = df_USA_Single_Columns_for_2017_2018_2019_2020['Month'] + '-' + df_USA_Single_Columns_for_2017_2018_2019_2020['Year']
        df_USA_Single_Columns_for_2017_2018_2019_2020.drop(columns = ['Date', 'Month', 'Year'], inplace=True)
        df_USA_Single_Columns_for_2017_2018_2019_2020['REF_DATE'] = pd.to_datetime(df_USA_Single_Columns_for_2017_2018_2019_2020['REF_DATE'])
        
        return df_USA_Single_Columns_for_2017_2018_2019_2020

def check_for_na(data, column_name):
    """Check if there are any missing values in the column_name of the data.
    Args:
        data::pandas_df: The data to check for missing values.
        column_name::str: The column to check for missing values.
    Returns:
        bool: True if there are missing values, False otherwise.
    """
    if data[column_name].isnull().values.any():
        return True
    else:
        return False
    
def check_data_type(data, column_name):
    """Check the data type of the column_name in the data.
    Args:
        data::pandas_df: The data to check for missing values.
        column_name::str: The column to check for missing values.
    Returns:
        str: The data type of the column_name in the data.
    """
    return column_name + " data type is: " + str(data[column_name].dtype)
    
def test_unit_dtype(data, column_name, dtype):
    """Check if the data type of the column_name in the data is equal to dtype.
    Args:
        data::pandas_df: The data to check for missing values.
        column_name::str: The column to check for missing values.
        dtype::str: The data type to check for.
    Returns:
        bool: True if the data type of the column_name in the data is equal to dtype, False otherwise.
    """ 
    return data[column_name].dtype == dtype

def test_unit_less_than_or_greater_than(data, column_name, value, gt_lt):
    """Check if the column_name in the data is less than or greater than value.
    Args:
        data::pandas_df: The data to check for missing values.
        column_name::str: The column to check for missing values.
        value::int: The value to check for.
        gt_lt::str: The condition to check for.
    Returns:
        bool: True if any of the column_name in the data is less than or greater than value, False otherwise.
    """
    if gt_lt == "less_than":
        return any(data[column_name] < value)
    else:
        return any(data[column_name] > value)
    
def test_unit_between(data, column_name, value1, value2):
    """Unit test to check if the column_name in the data is between value1 and value2.
    Args:
        data::pandas_df: The data to check for missing values.
        column_name::str: The column to check for missing values.
        value1::int: The first value to check for.
        value2::int: The second value to check for.
    Returns:
        bool: True if all of the column_name in the data is between value1 and value2, False otherwise.
    """
    return all((data[column_name] > value1) & (data[column_name] < value2))

def save_data(data, output_filepath):
    """Save the data to the output_filepath.
    Args:
        data::pandas_df: The data to save.
        output_filepath::str: The path to save the data.
    Returns:
        None
    """
    data.to_csv(output_filepath, index=False)
    return None

def look_at_missing_values(data, feature):
    """Look at missing values in the data.
    Args:
        data::pandas_df: The data to look at missing values.
    Returns:
        None
    """
    missing_data = data[data[feature].isnull()]
    print('Missing data: ', missing_data)
    return None

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
    