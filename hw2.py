from sklearn.linear_model import LogisticRegression
import pandas
import sys


#Written by Mert KAYA - 161101061 - 8 July 2020
#There is a code part commented in main method to print depth limited ID3

#setted globaly for dynamicprogramming
#attribute_keys holds names of all columns except target
global attribute_keys
global df_attributes
global prediction


#node for ID3
#name : holds a string array for identification of the node
#clf is only for the case that depth limit is reached
#clf : holds a Logistic Regression Predictor
class node(object):
    def __init__(self, name):
        self.name = name
        self.clf = None
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    def returnChildrenArray(self):
        return list(self.children)

#node for Naive bayes
#holds values as [ [attribute_name] [attribute value] [target value] [CalculatedProbability] ]
class naiveBayesNode(object):
    def __init__(self,values):
        self.values = values
        self.clf = None


#Returns true if S dataset has all 1 or all 0
def isAllPosorNeg(S):
    target_index = len(S.columns)-1
    target_name = S.columns[target_index]
    unique_val = S[target_name].unique()
    if(len(unique_val) != 2):
        return True
    else:
        return False 

#Setter method for globalvariable attribute_keys
#attribute_keys holds names of all columns except target
def create_attribute_array(df):
    global attribute_keys
    keys_array = list(df.keys())
    keys_array.pop(-1)
    attribute_keys = keys_array
    return


#Start of ID3
#Recursive ID3 algorithm implementation
#depth should be given 0 on all calls
#S: dataset , target : name of target attribute
#attributes : list of attributes which are used by ID3 for that call
def ID3(S,target,attributes,depth,depth_limit):

    #Can be just target_name = target
    target_index = len(S.columns)-1
    target_name = S.columns[target_index]

    #Initilazing the node which we will return
    returned_node = node("")

    #If depthlimit is reached
    if((int(depth) >= int(depth_limit)) and (depth_limit != -1)):

        #Check for allposorneg to not create a LogisticRegressionPredictor which only will return one target
        if(isAllPosorNeg(S)):
            #Nodes are created by various types 
            #If a node has target value initialized like below
            returned_node.name = "target",":",str(S.values[-1][-1])
            return returned_node
        
        #Splitting data into train and test where X:train y:test
        df = S.copy()
        y = df.iloc[:,-1]
        X = df.drop(df.columns[-1], axis=1)
        
        #Creating logisticregression predictor from sklearn
        clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None).fit(X,y)
        
        #If a node has target value initialized like below
        #node.clf holds the classifier
        returned_node.name = "clf"+":"
        returned_node.clf = clf
        return returned_node
    #End of depth limit part

    #Coded accordingly to the algorithm created by Ross Quinlan

    #If all elements are true or false return a node with targetlabel
    if(isAllPosorNeg(S)):
        returned_node.name = "target",":",str(S.values[-1][-1])
        return returned_node

    #If there are no attributes to check return node with most common value in S
    if(len(attributes) == 0):
        #Finding most common target value in S
        var =  S[target_name].value_counts().sort_values(ascending=False).keys()
        returned_node.name = "target",":",str(var[0])
        return returned_node

    # A <- Attribute that best classifies the S
    # that attribute can be found by finding the biggest gain
    # Gain(S,attributes) method returns that attribute 
    A = Gain(S,attributes)


    #Created root node with best attribute
    root_node = node("")
    root_node.name = ("root",":",A)

    #Each value A can take is found
    unique_val_array_root = findUniqueArray(S,A)

    #For each value A can take
    for value in unique_val_array_root:
        
        #Create children nodes with those values
        #Node is created by [[parent_name][:][value]]
        child = node((A,":",value))

        #Adding that child to A node
        root_node.add_child(child)

        #Svi = subset of S which has attribute[A] == value
        svi = getSvi(S,A,value)

        #If no element of S has attribute[A] == value 
        #Add a branch with label as most common target in S
        if(len(svi) == 0):
            returned_node = node("")
            #val = most common target in S
            val = getReturnedNodeValue(S,target_name)
            child2 = node(("target",":",str(val)))
            child.add_child(child2)            

        #Svi is not empty
        else:
            
            #get all attributes
            attributes_temp = attributes.copy()
            
            #Remove A from attribute list because it is alreadt checked
            attributes_temp.remove(A)

            #Recursively call ID3 with Svi and depth+1
            var = ID3(svi,target_name,attributes_temp,depth+1,depth_limit)

            #If clf is found
            if(type(var) == type(tuple)):
                #Node name = [["clf"] [":"]]
                returned_node.name = "" + var[0] + var[1]
                #Node.clf = returned clf
                returned_node.clf = var[2]
            
            #Nodes without clf
            else:
                #Var is a node gathered from recursive calls
                returned_node = var 

            #Add this node as a child to child
            #child = node with child of A
            child2 = returned_node
            child.add_child(child2)
            
    #Return the mainroot node
    return root_node
#End of ID3


#Return most common target value in S
def getReturnedNodeValue(S,target_name):
    temp_arr = S[target_name].value_counts().sort_values(ascending=False).keys()
    return temp_arr[0]


#Return subset of S which has attribute[A] = value
def getSvi(S,A,value):
    unique_df = S.loc[S[A] == value]
    return unique_df


#If attribute does not contain a key return true
def doesNotContain(attribute,key):
    for item in attribute:
        if(item == key):
            return False
    return True

#Find unique values of A attribute
#Which values A can get.
def findUniqueArray(S,A):
    if(len(A) == 0 or len(S) == 0):
        return []
    if(list(S[A]) == []):
        return []
    return list(S[A].unique())


#Start of Gain
#Returns the attribute which has the biggest gain in given df
def Gain(df,attribute):

    keys_array = df.keys()
    #Entropy of the df
    entropy_s = find_Entropy(df)

    max_gain = 0
    max_feature_val = ""
    
    #For each attribute
    for key in keys_array:

        #Not the target
        if(key == keys_array[-1]):
            break

        #If key is not in our attribute list continue
        if(doesNotContain(attribute,key)):
            continue

        #Unique values of attribute[key]
        unique_values = df[key].unique()
        gain = 0

        #For each unique value of key
        for unique_val in unique_values:
            
            #Unique_df contains all of the rows which has df[key] == unique_value
            unique_df = df.loc[df[key] == unique_val]
            
            #Gain for attribute is calculated by sum for all uniqs (size of uniqdf/df) * (entropy of unique df)
            gain += ((len(unique_df)/len(df))*(find_Entropy(unique_df)))

        #Gain is entropy of df - (sum of gain_values of attribute)
        gain = entropy_s - gain

        #Find the max_gain attribute
        if(gain > max_gain):
            max_gain = gain
            max_feature_val = key

    return max_feature_val
#End of Gain

#To find log without math library
def ln(x):
    n = 2147483647
    return n * ((x ** (1/n)) - 1)

#Calculating the log
def log(x,base):
    if(x == 0 & base == 0):
        return 0
    return ln(x)/ln(base)


#Returns Entropy of a given dataset
def find_Entropy(dataset):

    #Targets_df contains only target values of df
    targets_df = dataset.iloc[:,-1]
    
    #p_data[0] == number of 1's in target
    #p_data[1] == number of 0's in target
    p_data = targets_df.value_counts()   

    #In the case of 0 entropy
    if(len(p_data) != 2):
        return 0
    #In the case of 0 entropy
    if(p_data[0] == 0 | p_data[1]==0):
        return 0
    
    #Entropy can be calculated by -((numberof0s/sizeofdf)*(log:base2(numberof0s/sizeofdf))-((numberof1s/sizeofdf)*(log:base2(numberof1s/sizeofdf))
    num = (-(p_data[0]/len(dataset))*(log(p_data[0]/len(dataset),2)))-((p_data[1]/len(dataset))*(log(p_data[1]/len(dataset),2)))
    return num


#return value of an attribute in a given list of values with attribute name
def getVal(att,values):
    global attribute_keys
    attr_index = attribute_keys.index(att)
    return values[attr_index]

#To get prediction from ID3 
def predict(root_node,value_arr,text):
    target = ""
    
    #getting childs of root
    child_arr = root_node.returnChildrenArray()
    

    #global variable to hold prediction
    global prediction

    #If root is a logregclassifier
    if("clf" in root_node.name):
        #return the prediction gathered from predictor
        prediction = root_node.clf.predict([value_arr])
        return

    #Datas without proper attributes
    if(len(child_arr) == 0):
        return

    #If node is a "between node" which has only attribute name
    if(len(child_arr) == 1 and "target" not in str(child_arr[0].name[0])):

        #recursively calling child of that attribute
        #parent node : F1 -->  child node : F1=2
        predict(child_arr[0],value_arr,text)
        return

    #If a target node is reached which can only be reached from correct parent node which will get checked in if statement below
    if("target" in child_arr[0].name[0]):

        #Text is returned to print nicely
        #Returned text value can be used to determine the path it has taken
        text += "result = " + str(child_arr[0].name[2])
        target = str(child_arr[0].name[2])

        #global value prediction is setted 
        prediction = target
        return text
    
    #Checking if node is main root or parent of a subroot
    #Root in name means node with only attribute name 
    #Child of this node will contain attribute_name-->value
    if("root" in child_arr[0].name[0] or "root" in root_node.name):
        #For each child node with values
        for child in child_arr:
            #If the path of value for the value wanted to be predicted is found
            if(str(child.name[2]) == str(getVal(child.name[0],value_arr))):
                text = text + "" + str(child.name[0]) + " =" + str(child.name[2]) + ","
                #recursive call from that branch
                predict(child,value_arr,text)
                break


#Start KFold ID3
#Returns accuracy for 5fold ID3withdepthlimit and logreg
def K_fold(df,target,attr,depth,depthlimit):

    #If df is smaller than 5 take 1 row as test data
    if(len(df) < 5):
        test_data = df[0:1]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[0:1],inplace=True)
        train_data = temp_df
        #return how many is accurately predicted / size of test_df
        return(findTrueNumber(train_data,test_data,target,attr,depth,depthlimit)/len(test_data))


    start_index = 0
    #How many elements are in each fold
    added_number = int(len(df)/5)
    end_index = start_index + added_number
    sum_true = 0 

    #Iterate 5 times
    for i in range(1,6):
        if(i == 5):
            end_index = len(df)

        #Data is splitted into 1 / 4 with calculated indexes
        #Each iteration will be made with a different testfold
        row_count = df.shape[0]
        test_data = df[start_index:end_index]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[start_index:end_index],inplace=True)
        start_index = start_index + added_number
        end_index = start_index + added_number
        train_data = temp_df

        #Sum of correctly predicted times is added
        sum_true = sum_true + findTrueNumber(train_data,test_data,target,attr,depth,depthlimit)
    
    # In the case of 1000 elements in df all have been as test data and correct ones are counted
    #accuracy = allcorrectones/all
    return (str(sum_true/len(df)))

#End FoldID3


#Returns the number of correct predictions
def findTrueNumberNaiveBayes(train_data,test_data):

    #NaiveBayes model is created
    model_array = returnNaiveBayesModel(train_data)
    true_val_num = 0
    #Target name is gathered
    target_index = len(train_data.columns)-1
    target_name = train_data.columns[target_index]

    #For each testdata
    for i in range(len(test_data.values)-1):

        #Given value
        yi = str(test_data[target_name].values[i])
        
        #Get the values for prediction
        values = getRowbyIndex(test_data,i)

        #Predicting the value
        predicted = findandCalcResult(model_array,values,target_name)
        y_est = predicted

        #If correct add 1
        if(str(yi) == str(y_est)):
            true_val_num = true_val_num + 1 
        
    return true_val_num


#Predict method for NB model
#Returns Target 1 or 0 for NaiveBayes
#Start of findandCalcResult
def findandCalcResult(model_array,values,target_name):
    global attribute_keys

    index = 0
    zero_probability = 1
    one_probability = 1
    
    #For each value in values such as [1,1,1,1,2,3]
    for val in values:
        
        #Get the attribute by index
        attr_name = list(attribute_keys)[index]
        index = index + 1
        
        found_control0 = False
        found_control1 = False
        
        #Check if naivebayes model has probability for that attribute and value
        for submodel in model_array:
            if(str(submodel[0]) == str(attr_name)):
                if(submodel[1] == val):
                    if(submodel[2] == 0):
                        found_control0 = True
                        #Multiply probability of zero with found probability in NBmodel
                        zero_probability = zero_probability*submodel[3]
                    else:
                        found_control1 = True
                        #Multiply probability of one with found probability in NBmodel
                        one_probability = one_probability*submodel[3]
        
        #If none found it means there are no occurences of them in training
        #Probability is set to 0 because 0 * other prob. = 0

        if(found_control0 == False):
            zero_probability = 0
        if(found_control1 == False):
            one_probability = 0

    #Return target with bigger probability as prediction
    if(one_probability > zero_probability):
        return 1
    else:
        return 0
#End of findandCalcResult



#Number of correct predictions for ID3
#Start of findTrueNumber
def findTrueNumber(train_data,test_data,target,attr,depth,depthlimit):
    global prediction

    #ID3 is created
    root = ID3(train_data,target,attr,0,depthlimit)

    true_val_num = 0
    for i in range(len(test_data.values)-1):
        yi = str(test_data[target].values[i])
        #values = test_data.loc[[i]]
        values = getRowbyIndex(test_data,i)

        #predictmethod assigns global prediction value
        text = predict(root,values,"")
        global prediction
        y_est = prediction

        #If correct numberofcorrectpredictions + 1
        if(str(yi) == str(y_est)):
            true_val_num = true_val_num + 1 
        
    return true_val_num
#Endtart of findTrueNumber


#Returns row by given index
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
            

#Prints tree with indentation recursively
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

#Start of returnNaiveBayesModel
#Creates and returns an array with 
# probabilitys of all combinations of each attribute,target in training set
def returnNaiveBayesModel(df):

    #Attribute names are gathered
    keys_array = list(df.keys())
    Model_array = []
    target = keys_array.pop(-1)
    attribute_array = keys_array 
    
    #For each attribute
    for att in attribute_array: 

        #For each value an attribute can get
        uniques = findUniqueArray(df,att)
        for uniq in uniques:
            #For each target in dataset
            unique_targets = findUniqueArray(df,target)
            for uniq_target in unique_targets:
                model = []
                model.append(str(att))
                model.append(str(uniq))
                model.append(str(uniq_target))
                model.append(calcProbability(df,att,uniq,uniq_target))
                #Each model is in form : [[attribute name][value of that attribute][target of that attribute][probability of this]]
                Model_array.append(model)
    return Model_array

#Returns probability of given attribute, its value and target 
#Start of calcProbability
def calcProbability(df,att,uniq,uniq_target):
    all_count = 0
    row_count = 0

    #For each row
    for row in df.itertuples():
        index = attribute_keys.index(att)
        #If attribute is not in dataset
        if(index == -1):
            return 0
        
        #rows of df has 'index' as firstcolumn
        index = index+1

        #all_count = all occurences of att[uniq]
        #row_count = all occurences of att[uniq] == uniq_target
        if(int(row[index]) == int(uniq)):
            all_count = all_count + 1
            if(int(row[-1]) == int(uniq_target)):
                row_count = row_count + 1  

    return (row_count/all_count)
#End of calcProbability



#Returns accuracy of NaiveBayes with 5fold 
def kFoldNaiveBayes(df):
    
    #To iterate folds
    start_index = 0
    added_number = int(len(df)/5)
    end_index = start_index + added_number
    sum_true = 0 

    #If last fold
    for i in range(1,6):
        if(i == 5):
            end_index = len(df)


        #Traindata and test data is separated by indexes
        #Each iteration has different testfold with this
        test_data = df[start_index:end_index]
        temp_df = df.copy()
        temp_df.drop(temp_df.index[start_index:end_index],inplace=True)
        start_index = start_index + added_number
        end_index = start_index + added_number
        train_data = temp_df
        
        #Count the correctpredictions in this iteration
        sum_true = sum_true + findTrueNumberNaiveBayes(train_data,test_data)

    #accuracy = sumAllCorrectPredictions / sizeofAlldataset
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
    
    #Attributes are found and globalvariable of its is setted
    target_index = len(df.columns)-1
    target_name = df.columns[target_index]
    df_without_index = df.drop(columns=[target_name])
    df = df.drop(df.columns[0], axis=1)
    attributes = df.keys()
    attributes = list(attributes[:-1])
    global df_attributes
    df_attributes = attributes

    #globalvariable for attribute keys is set
    create_attribute_array(df)


    #Default depth is -1 which means no depthlimit
    #LogisticRegression will not be used with this
    depth_value = -1

    #If depth_limit is given 
    if(len(sys.argv) > 3):
        #root = ID3(df,target_name,attributes,0,int(sys.argv[3]))
        depth_value = int(sys.argv[3])
    else:
        depth_value = -1
        #root = ID3(df,target_name,attributes,0,-1)

    #To print the tree uncomment the above lines for root and uncomment below
    #printTree(root,0)
 
    
    print("DTLog: ",K_fold(df,target_name,attributes,0,depth_value))
    print("NB: ",kFoldNaiveBayes(df))

if __name__ == "__main__":
    main()