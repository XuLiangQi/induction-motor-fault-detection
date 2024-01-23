import pandas as pd
import glob 

def load_data():
    normal = glob.glob('app/data/normal/normal/' + '*.csv')
    imbalance6g = glob.glob('app/data/imbalance/imbalance/6g/' + '*.csv')
    imbalance10g = glob.glob('app/data/imbalance/imbalance/10g/' + '*.csv')
    imbalance15g = glob.glob('app/data/imbalance/imbalance/15g/' + '*.csv')
    imbalance20g = glob.glob('app/data/imbalance/imbalance/20g/' + '*.csv')
    imbalance25g = glob.glob('app/data/imbalance/imbalance/25g/' + '*.csv')
    imbalance30g = glob.glob('app/data/imbalance/imbalance/30g/' + '*.csv')
    imbalance35g = glob.glob('app/data/imbalance/imbalance/35g/' + '*.csv')

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
    data_35g = read_csv_from_path(imbalance35g)

    return data_n, data_6g, data_10g, data_15g, data_20g, data_25g, data_30g, data_35g