from sklearn.linear_model import LogisticRegression
import pandas
import sys

global attribute_keys
global df_attributes
global prediction
class node(object):
    def __init__(self, name):
        self.name = name
        self.clf = None
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    def returnChildrenArray(self):
        return list(self.children)

class naiveBayesNode(object):
    def __init__(self,values):
        self.values = values
        self.clf = None

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


def ID3(S,target,attributes,depth,depth_limit):
    target_index = len(S.columns)-1
    target_name = S.columns[target_index]
    returned_node = node("")
    if(int(depth) >= int(depth_limit)):
        if(isAllPosorNeg(S)):
            returned_node.name = "target",":",str(S.values[-1][-1])
            return returned_node
        df = S.copy()
        y = df.iloc[:,-1]
        X = df.drop(df.columns[-1], axis=1)
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None).fit(X,y)
        returned_node.name = "clf"+":"
        returned_node.clf = clf
        return returned_node

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
            var = ID3(svi,target_name,attributes_temp,depth+1,depth_limit)
            if(type(var) == type(tuple)):
                returned_node.name = "" + var[0] + var[1]
                returned_node.clf = var[2]
            else:
                returned_node = var 
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
    global prediction
    #Attribute child.
    if("clf" in root_node.name):
        prediction = root_node.clf.predict([value_arr])
        return
    if(len(child_arr) == 1 and "target" not in str(child_arr[0].name[0])):
        predict(child_arr[0],value_arr,text)
        return

    if("target" in child_arr[0].name[0]):
        text += "result = " + str(child_arr[0].name[2])
        target = str(child_arr[0].name[2])
        prediction = target
        return text
    
    #main root or subroot
    if("root" in child_arr[0].name[0] or "root" in root_node.name):
        for child in child_arr:
            if(str(child.name[2]) == str(getVal(child.name[0],value_arr))):
                text = text + "" + str(child.name[0]) + " =" + str(child.name[2]) + ","
                predict(child,value_arr,text)
                break


def K_fold(df,target,attr,depth,depthlimit):
    if(len(df) < 5):
        test_data = df[0:1]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[0:1],inplace=True)
        train_data = temp_df
        return(findTrueNumber(train_data,test_data,target,attr,depth,depthlimit)/len(df))

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
        sum_true = sum_true + findTrueNumber(train_data,test_data,target,attr,depth,depthlimit)
    return (str(sum_true/len(df)))

def findTrueNumberNaiveBayes(train_data,test_data):
    model_array = returnNaiveBayesModel(train_data)
    true_val_num = 0
    target_index = len(train_data.columns)-1
    target_name = train_data.columns[target_index]
    for i in range(len(test_data.values)-1):
        yi = str(test_data[target_name].values[i])
        #values = test_data.loc[[i]]
        values = getRowbyIndex(test_data,i)
        predicted = findandCalcResult(model_array,values,target_name)
        y_est = predicted
        if(str(yi) == str(y_est)):
            true_val_num = true_val_num + 1 
        
    return true_val_num

#returns target 1 or 0
def findandCalcResult(model_array,values,target_name):
    global attribute_keys
    index = 0
    zero_probability = 1
    one_probability = 1
    for val in values:
        attr_name = list(attribute_keys)[index]
        index = index + 1
        found_control0 = False
        found_control1 = False
        for submodel in model_array:
            if(str(submodel[0]) == str(attr_name)):
                if(submodel[1] == val):
                    if(submodel[2] == 0):
                        found_control0 = True
                        zero_probability = zero_probability*submodel[3]
                    else:
                        found_control1 = True
                        one_probability = one_probability*submodel[3]
        if(found_control0 == False):
            zero_probability = 0
        if(found_control1 == False):
            one_probability = 0

    if(one_probability > zero_probability):
        return 1
    else:
        return 0



def findTrueNumber(train_data,test_data,target,attr,depth,depthlimit):
    global prediction

    root = ID3(train_data,target,attr,0,depthlimit)

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
    for i in range(number):
        text = "   " + text
    for child in var:
        print(text,"ch",number," : ",child.name)
        printTree(child,number+1)
             
def returnNaiveBayesModel(df):
    #if not in model array return 0
    keys_array = list(df.keys())
    Model_array = []
    target = keys_array.pop(-1)
    attribute_array = keys_array 
    for att in attribute_array: 
        uniques = findUniqueArray(df,att)
        for uniq in uniques:
            unique_targets = findUniqueArray(df,target)
            for uniq_target in unique_targets:
                model = []
                model.append(str(att))
                model.append(str(uniq))
                model.append(str(uniq_target))
                model.append(calcProbability(df,att,uniq,uniq_target))
                Model_array.append(model)
    return Model_array

def calcProbability(df,att,uniq,uniq_target):
    all_count = 0
    row_count = 0
    for row in df.itertuples():
        index = attribute_keys.index(att)
        if(index == -1):
            return 0
        
        #rows of df has 'index' as firstcolumn
        index = index+1

        if(int(row[index]) == int(uniq)):
            all_count = all_count + 1
            if(int(row[-1]) == int(uniq_target)):
                row_count = row_count + 1  
    return (row_count/all_count)

def kFoldNaiveBayes(df):
    
    start_index = 0
    added_number = int(len(df)/5)
    end_index = start_index + added_number
    sum_true = 0 

    for i in range(1,6):
        if(i == 5):
            end_index = len(df)

    
        test_data = df[start_index:end_index]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[start_index:end_index],inplace=True)
        start_index = start_index + added_number
        end_index = start_index + added_number
        train_data = temp_df
        sum_true = sum_true + findTrueNumberNaiveBayes(train_data,test_data)
    return (sum_true/len(df))

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

    create_attribute_array(df)
    if(len(sys.argv) > 3):
        root = ID3(df,target_name,attributes,0,int(sys.argv[3]))
    else:
        root = ID3(df,target_name,attributes,0,-1)

    #printTree(root,0)
    #chr_arr = root.returnChildrenArray()
    #predict(root,[2],"")
    #getSvi(df,"F1",1)
    
    print(K_fold(df,target_name,attributes,0,int(sys.argv[3])))
    model_array = returnNaiveBayesModel(df)
    print(kFoldNaiveBayes(df))

if __name__ == "__main__":
    main()