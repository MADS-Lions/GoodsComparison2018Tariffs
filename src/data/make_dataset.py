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
def load_data(input_filepath):
    """Load data from input_filepath"""
    data = pd.read_csv(input_filepath)
    return data

def manipulate_data(data, which_data_file):
    if which_data_file == "Canadian":
        Canadian_CPI_dataset = data.copy()
        Canadian_CPI_dataset = Canadian_CPI_dataset.groupby(['Products and product groups', 'REF_DATE']).mean()    
        Canadian_CPI_dataset.reset_index(inplace=True)
        return Canadian_CPI_dataset
    else:
        USA_CPI_dataset = data.copy()
        USA_CPI_dataset = USA_CPI_dataset[USA_CPI_dataset['DATA TYPE'] == 'SEASONALLY ADJUSTED INDEX']
        USA_CPI_dataset.drop(columns = ['ITEM', 'series id', 'DATA TYPE'], inplace=True)
        USA_CPI_dataset = USA_CPI_dataset.groupby(['TITLE', 'YEAR']).mean()
        return USA_CPI_dataset


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
    