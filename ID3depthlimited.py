import pandas
import sys

global attribute_keys
global df_attributes
global prediction
class node(object):
    def __init__(self, name):
        self.name = name
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    def returnChildrenArray(self):
        return list(self.children)


def isAllPosorNeg(S):
    target_index = len(S.columns)-1
    target_name = S.columns[target_index]
    unique_val = S[target_name].unique()
    if(len(unique_val) != 2):
        return True
    else:
        return False 

def create_attribute_array(df):
    global attribute_keys
    keys_array = list(df.keys())
    keys_array.pop(-1)
    attribute_keys = keys_array
    return


def ID3(S,target,attributes):
    target_index = len(S.columns)-1
    target_name = S.columns[target_index]
    returned_node = node("")

    if(isAllPosorNeg(S)):
        returned_node.name = "target",":",str(S.values[-1][-1])
        return returned_node

    if(len(attributes) == 0):
        var =  S[target_name].value_counts().sort_values(ascending=False).keys()
        returned_node.name = "target",":",str(var[0])
        return returned_node
    
    A = Gain(S,attributes)
    root_node = node("")
    root_node.name = ("root",":",A)
    #print("A : ",A)
    unique_val_array_root = findUniqueArray(S,A)
    for value in unique_val_array_root:

        child = node((A,":",value))
        root_node.add_child(child)
        svi = getSvi(S,A,value)

        if(len(svi) == 0):
            returned_node = node("")
            val = getReturnedNodeValue(S,target_name)
            child2 = node(("target",":",str(val)))
            child.add_child(child2)            
        else:
            attributes_temp = attributes.copy()
            attributes_temp.remove(A)
            returned_node = ID3(svi,target_name,attributes_temp)
            child2 = returned_node
            child.add_child(child2)

            
    return root_node

        #returned_node = node()
        #returned_node.name = S.values[-1][-1]
        #return returned_node

def getReturnedNodeValue(S,target_name):
    temp_arr = S[target_name].value_counts().sort_values(ascending=False).keys()
    return temp_arr[0]


def getSvi(S,A,value):
    unique_df = S.loc[S[A] == value]
    return unique_df

def doesNotContain(attribute,key):
    for item in attribute:
        if(item == key):
            return False
    return True

def findUniqueArray(S,A):
    if(list(S[A]) == []):
        return []
    return list(S[A].unique())

def Gain(df,attribute):
    keys_array = df.keys()
    entropy_s = find_Entropy(df)
    max_gain = 0
    max_feature_val = ""
    for key in keys_array:
        if(key == keys_array[-1]):
            break
        if(doesNotContain(attribute,key)):
            continue
        unique_values = df[key].unique()
        gain = 0
        for unique_val in unique_values:
            #check this one
            unique_df = df.loc[df[key] == unique_val]
            #if unique_df = empty continue? or not possible?
            gain += ((len(unique_df)/len(df))*(find_Entropy(unique_df)))
        gain = entropy_s - gain
        if(gain > max_gain):
            max_gain = gain
            max_feature_val = key
    return max_feature_val


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


def getVal(att,values):
    global attribute_keys
    attr_index = attribute_keys.index(att)
    return values[attr_index]

def predict(root_node,value_arr,text):
    target = ""
    child_arr = root_node.returnChildrenArray()
    #Attribute child.
    if(len(child_arr) == 1 and "target" not in str(child_arr[0].name[0])):
        predict(child_arr[0],value_arr,text)
        return

    if("target" in child_arr[0].name[0]):
        text += "result = " + str(child_arr[0].name[2])
        target = str(child_arr[0].name[2])
        global prediction
        prediction = target
        #print(text)
        return text
    
    #main root or subroot
    if("root" in child_arr[0].name[0] or "root" in root_node.name):
        for child in child_arr:
            if(str(child.name[2]) == str(getVal(child.name[0],value_arr))):
                text = text + "" + str(child.name[0]) + " =" + str(child.name[2]) + ","
                predict(child,value_arr,text)
                break

    
def K_fold(df,target,attr):
    if(len(df) < 5):
        row_count = df.shape[0]
        test_data = df[0:1]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[0:1],inplace=True)
        train_data = temp_df
        print(findTrueNumber(train_data,test_data,target,attr)/len(df))
        return

    start_index = 0
    added_number = int(len(df)/5)
    end_index = start_index + added_number
    sum_true = 0 
    for i in range(1,6):
        if(i == 5):
            end_index = len(df)

        row_count = df.shape[0]
        test_data = df[start_index:end_index]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[start_index:end_index],inplace=True)
        start_index = start_index + added_number
        end_index = start_index + added_number
        train_data = temp_df
        sum_true = sum_true + findTrueNumber(train_data,test_data,target,attr)
    print(sum_true/len(df))

def findTrueNumber(train_data,test_data,target,attr):
    global prediction

    root = ID3(train_data,target,attr)

    true_val_num = 0
    for i in range(len(test_data.values)-1):
        yi = str(test_data[target].values[i])
        #values = test_data.loc[[i]]
        values = getRowbyIndex(test_data,i)

        #predictmethod assigns global prediction value
        var = predict(root,values,"")
        y_est = prediction
        if(str(yi) == str(y_est)):
            true_val_num = true_val_num + 1 
        
    return true_val_num


def getRowbyIndex(df,index):
    global attribute_keys
    count = 0
    returned_values = []
    for row in df.itertuples():
        if(count == index):
            for i in range(1,len(row)-1):
                returned_values.append(row[i])
            return returned_values
        count = count + 1
            

#def find_Entropy():
def printTree(root_node,number):
    var = root_node.returnChildrenArray()
    if(number == 0):
        print("root : ",root_node.name)
    text = ""
    for child in var:
        for i in range(number):
            text = "   " + text
        print(text,"ch",number," : ",child.name)
        printTree(child,number+1)
             


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
    attributes = df.keys()
    attributes = list(attributes[:-1])
    global df_attributes
    df_attributes = attributes
    printTree(ID3(df,target_name,attributes),0)
    print("------------------------------------")
    print("------------------------------------")
    print("------------------------------------")
    create_attribute_array(df)
    #root = ID3(df,target_name,attributes)
    #chr_arr = root.returnChildrenArray()
    #predict(root,[2],"")
    #getSvi(df,"F1",1)
    K_fold(df,target_name,attributes)
if __name__ == "__main__":
    main()