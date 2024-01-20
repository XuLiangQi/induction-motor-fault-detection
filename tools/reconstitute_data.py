import os
import glob
import signal
import pandas as pd

def add_label(path, label):
    for i in path:
        data = pd.read_csv(i, header=None)
        data[8] = label
        new_i = i[:5] + "reconstituted_data/" + i[5:]
        path_to_new_i = os.path.dirname(new_i)
        if not os.path.exists(path_to_new_i):
            os.makedirs(path_to_new_i)
        if not os.path.exists(new_i):
            data.to_csv(new_i, index=False)

def reconstitute_data_with_label():
    normalData = glob.glob('data/normal/' + '*.csv')
    imbalance6gData = glob.glob('data/imbalance/6g/' + '*.csv')
    imbalance10gData = glob.glob('data/imbalance/10g/' + '*.csv')
    imbalance15gData = glob.glob('data/imbalance/15g/' + '*.csv')
    imbalance20gData = glob.glob('data/imbalance/20g/' + '*.csv')
    imbalance25gData = glob.glob('data/imbalance/25g/' + '*.csv')
    imbalance30gData = glob.glob('data/imbalance/30g/' + '*.csv')
    imbalance35gData = glob.glob('data/imbalance/35g/' + '*.csv')

    add_label(normalData, 0)
    add_label(imbalance6gData, 1)
    add_label(imbalance10gData, 2)
    add_label(imbalance15gData, 3)
    add_label(imbalance20gData, 4)
    add_label(imbalance25gData, 5)
    add_label(imbalance30gData, 6)
    add_label(imbalance35gData, 7)


def reconstitute_data_by_splitting():
    def split_each_file(path):
        dataTrain = pd.DataFrame()
        dataTest = pd.DataFrame()
        dataVal = pd.DataFrame()

        total_num = len(path)
        idx = 0
        for i in path:
            if idx < round(total_num * 0.80, ndigits=None):
                data = pd.read_csv(i)
                dataTrain = pd.concat([dataTrain, data], ignore_index=True)
            elif idx < round(total_num * 0.90, ndigits=None):
                data = pd.read_csv(i)
                dataTest = pd.concat([dataTest, data], ignore_index=True)
            else:
                data = pd.read_csv(i)
                dataVal = pd.concat([dataVal, data], ignore_index=True)
            idx += 1
        return dataTrain, dataTest, dataVal

   
    dataTrain = pd.DataFrame()
    dataTest = pd.DataFrame()
    dataVal = pd.DataFrame()
    normTrain, normTest, normVal = split_each_file(glob.glob('data/reconstituted_data/normal/' + '*.csv'))
    g6Train, g6Test, g6Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/6g/' + '*.csv'))
    g10Train, g10Test, g10Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/10g/' + '*.csv'))
    g15Train, g15Test, g15Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/15g/' + '*.csv'))
    g20Train, g20Test, g20Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/20g/' + '*.csv'))
    g25Train, g25Test, g25Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/25g/' + '*.csv'))
    g30Train, g30Test, g30Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/30g/' + '*.csv'))
    g35Train, g35Test, g35Val = split_each_file(glob.glob('data/reconstituted_data/imbalance/35g/' + '*.csv'))
    dataTrain = pd.concat([dataTrain, normTrain, g6Train, g10Train, g15Train, g20Train, g25Train, g30Train, g35Train]
                          , ignore_index=True)
    dataTest = pd.concat([dataTest, normTest, g6Test, g10Test, g15Test, g20Test, g25Test, g30Test, g35Test]
                         , ignore_index=True)
    dataVal = pd.concat([dataVal, normVal, g6Val, g10Val, g15Val, g20Val, g25Val, g30Val, g35Val]
                        , ignore_index=True)
    
    dataTrain.to_csv('data/reconstituted_data/data_train.csv', index=False)
    dataTest.to_csv('data/reconstituted_data/data_test.csv', index=False)
    dataVal.to_csv('data/reconstituted_data/data_val.csv', index=False)

if __name__ == "__main__":
    def handler(signum, frame):
        raise TimeoutError("Timed out!")
    try:
        # Set the timeout to 10 seconds
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        # Get user input
        print(" ---------------------- WARNING -----------------------")
        print("This action will replace any already processed data files.")
        user_input = input("Do you want to proceed? (y/n): ")
        # Reset the alarm after successful input
        signal.alarm(0)
        # Check if the user wants to proceed
        if user_input.lower() == 'y':
            # Perform the desired action
            print("Proceeding with the action.")

            reconstitute_data_with_label()
            reconstitute_data_by_splitting()
        else:
            print("Operation canceled.")
    except TimeoutError:
        print("Timeout: No response received within 10 seconds. Operation canceled.")

