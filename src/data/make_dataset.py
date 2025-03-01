# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
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

def load_data(input_filepath, which_data_file):
    """Load data from input_filepath"""
    if which_data_file == '.csv':
        data = pd.read_csv(input_filepath)
    else:
        data = pd.read_excel(input_filepath)
    return data

def manipulate_data(data, which_data_file):
    if which_data_file == "Canada":
        Canadian_CPI_dataset = data.copy()
        Canadian_CPI_dataset['REF_DATE'] = pd.to_datetime(Canadian_CPI_dataset['REF_DATE'])
        Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['REF_DATE']>='2017-01-01')&(Canadian_CPI_dataset['REF_DATE']<='2019-12-31')]
        print('UOM has two unique values in the dataset: ', Canadian_CPI_dataset['UOM'].unique())
        Canadian_CPI_dataset = Canadian_CPI_dataset[(Canadian_CPI_dataset['UOM']=='2002=100')]
        print('UOM now only has 2002 value: ', Canadian_CPI_dataset['UOM'].unique())
        Canadian_CPI_dataset = Canadian_CPI_dataset[['Products and product groups', 'REF_DATE', 'VALUE']]
        Canadian_CPI_dataset = Canadian_CPI_dataset.groupby(['Products and product groups', 'REF_DATE']).mean()    
        Canadian_CPI_dataset.reset_index(inplace=True)
        Canadian_CPI_dataset['REF_DATE'] = pd.to_datetime(Canadian_CPI_dataset['REF_DATE'])
        return Canadian_CPI_dataset
    else:
        USA_CPI_dataset = data.copy()
        USA_CPI_dataset = USA_CPI_dataset[USA_CPI_dataset['DATA_TYPE'] == 'SEASONALLY ADJUSTED INDEX']
        USA_CPI_dataset = USA_CPI_dataset[(USA_CPI_dataset['YEAR']>=2017)&(USA_CPI_dataset['YEAR']<=2019)]
        USA_CPI_dataset.drop(columns = ['ITEM', 'seriesid', 'DATA_TYPE'], inplace=True)
        USA_CPI_dataset = USA_CPI_dataset.groupby(['TITLE', 'YEAR']).mean()
        USA_CPI_dataset_T = USA_CPI_dataset.T
        df_USA_Single_Columns_for_2017_2018_2019 = pd.DataFrame()
        
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
                column_to_add = pd.concat([dataframe_2017, dataframe_2018, dataframe_2019], axis=0)
                df_USA_Single_Columns_for_2017_2018_2019 = pd.concat([df_USA_Single_Columns_for_2017_2018_2019, column_to_add], axis=1)
        df_USA_Single_Columns_for_2017_2018_2019.reset_index(inplace=True)
        df_USA_Single_Columns_for_2017_2018_2019.rename(columns = {'index': 'Date'}, inplace=True)

        df_USA_Single_Columns_for_2017_2018_2019['Month'] = df_USA_Single_Columns_for_2017_2018_2019['Date'].str.split('_').str[0]

        df_USA_Single_Columns_for_2017_2018_2019['Year'] = df_USA_Single_Columns_for_2017_2018_2019['Date'].str.split('_').str[1]
        df_USA_Single_Columns_for_2017_2018_2019['REF_DATE'] = df_USA_Single_Columns_for_2017_2018_2019['Month'] + '-' + df_USA_Single_Columns_for_2017_2018_2019['Year']
        df_USA_Single_Columns_for_2017_2018_2019.drop(columns = ['Date', 'Month', 'Year'], inplace=True)
        df_USA_Single_Columns_for_2017_2018_2019['REF_DATE'] = pd.to_datetime(df_USA_Single_Columns_for_2017_2018_2019['REF_DATE'])
        df_USA_Single_Columns_for_2017_2018_2019.set_index('REF_DATE', inplace=True)
        return df_USA_Single_Columns_for_2017_2018_2019

def check_for_na(data, column_name):
    if data[column_name].isnull().values.any():
        return True
    else:
        return False
    
def check_data_type(data, column_name):
    return column_name + " data type is: " + str(data[column_name].dtype)
    
def test_unit_dtype(data, column_name, dtype): 
    return data[column_name].dtype == dtype

def test_unit_less_than_or_greater_than(data, column_name, value, gt_lt):
    if gt_lt == "less_than":
        return any(data[column_name] < value)
    else:
        return any(data[column_name] > value)
    
def test_unit_between(data, column_name, value1, value2):
    return all((data[column_name] > value1) & (data[column_name] < value2))

def save_data(data, output_filepath):
    data.to_csv(output_filepath, index=False)
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
    