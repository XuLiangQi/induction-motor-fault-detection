import pandas as pd
import glob 

def read_csv_from_path(path_names: str) -> pd.DataFrame:
    '''Read the CSV file from the given path
    Parameters:
    ----------
    
    path_names: str
        The path that leads to the targeting CSV file. 
    
    Returns:
    -------

    data_n: pd.DataFrame
        The data loaded from the CSV file from the provided path.
    '''
    data_n = pd.DataFrame()
    for i in path_names:
        low_data = pd.read_csv(i,header=None)
        data_n = pd.concat([data_n,low_data],ignore_index=True, axis=0)
    return data_n
def load_data():
    '''Load the data (csv files) from all the sub directories.
    
    returns:
    -------

    data: pd.DataFrame
        The DataFrame read from all the CSV files under each path.
    '''
    normal = glob.glob('app/data/normal/normal/' + '*.csv')
    imbalance6g = glob.glob('app/data/imbalance/imbalance/6g/' + '*.csv')
    imbalance10g = glob.glob('app/data/imbalance/imbalance/10g/' + '*.csv')
    imbalance15g = glob.glob('app/data/imbalance/imbalance/15g/' + '*.csv')
    imbalance20g = glob.glob('app/data/imbalance/imbalance/20g/' + '*.csv')
    imbalance25g = glob.glob('app/data/imbalance/imbalance/25g/' + '*.csv')
    imbalance30g = glob.glob('app/data/imbalance/imbalance/30g/' + '*.csv')

    def read_csv_from_path(path_names):
        data_n = pd.DataFrame()
        for i in path_names:
            low_data = pd.read_csv(i,header=None)
            data_n = pd.concat([data_n,low_data],ignore_index=True, axis=0)
        return data_n
    
    data_n = read_csv_from_path(normal)
    data_6g = read_csv_from_path(imbalance6g)
    data_10g = read_csv_from_path(imbalance10g)
    data_15g = read_csv_from_path(imbalance15g)
    data_20g = read_csv_from_path(imbalance20g)
    data_25g = read_csv_from_path(imbalance25g)
    data_30g = read_csv_from_path(imbalance30g)

    return data_n, data_6g, data_10g, data_15g, data_20g, data_25g, data_30g