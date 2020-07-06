import pandas
import sys

def Gain(df,attr):
    #targets_df = df.iloc[:,-1]
    keys_array = df.keys()
    #var_containing_df = df[df.isin(2).any(1)]
    #df2 = df.loc[(df.ix[:,0] == 2)]
    entropy_s = find_Entropy(df)
    max_gain = 0
    max_feature_val = ""
    for key in keys_array:
        if(key == keys_array[-1]):
            break
        unique_values = df[key].unique()
        #print(unique_values)
        gain = 0
        #print("kontrol",val)
        for unique_val in unique_values:
            #print("key:",key,"---","unique : ",unique_val)
            unique_df = df.loc[df[key] == unique_val]
            print("entropy of : ",key,"val:",unique_val,find_Entropy(unique_df))
            gain += ((len(unique_df)/len(df))*(find_Entropy(unique_df)))
        print("entropy1",entropy_s)
        print("mg1",max_gain)
        gain = entropy_s - gain
        if(gain > max_gain):
            max_gain = gain
            max_feature_val = key
    print("mgain : ",max_gain,"---feature : ",max_feature_val)
    #print("entropy1",val)
def ln(x):
    n = 2147483647
    return n * ((x ** (1/n)) - 1)
def log(x,base):
    if(x == 0 & base == 0):
        return 0
    return ln(x)/ln(base)

def find_Entropy(dataset):
    skip_first = True
    entropy = 0
    targets_df = dataset.iloc[:,-1]
    p_data = targets_df.value_counts() 
    
    #In the case of 0 entropy
    if(len(p_data) != 2):
        return 0
    #In the case of 0 entropy
    if(p_data[0] == 0 | p_data[1]==0):
        return 0

    num = (-(p_data[0]/len(dataset))*(log(p_data[0]/len(dataset),2)))-((p_data[1]/len(dataset))*(log(p_data[1]/len(dataset),2)))
    return num

#def split_data_set(df):
#    dataset_array = []
#    print(dataset_array)
    

#def find_Entropy():

def main():
    #Model and dataset are gathered from the arguments
    dataset_name = sys.argv[1]

    #Reading dataset with pandas
    df = pandas.DataFrame(pandas.read_csv(dataset_name))

    #Create a splitted dataset for values of features and obtain the feature_value_size
    target_index = len(df.columns)-1
    target_name = df.columns[target_index]
    df_without_index = df.drop(columns=[target_name])
    df = df.drop(df.columns[0], axis=1)
    #print(df_without_index)
    #print(find_Entropy(df))
    #print(dataset_array)
    Gain(df,0)




if __name__ == "__main__":
    main()