import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def getdata(filepath): #读取sogou的数据
    sep_sign = ","
    with open(file_path, 'r') as file:
        lines = file.readlines()
        comma_count = sum([line.count(',') for line in lines[:5]])
        semicolon_count = sum([line.count(';') for line in lines[:5]])
        
    if comma_count > semicolon_count:
        sep_sign  = ","
    else:
        sep_sign  = ";"
    df = pd.read_csv(filepath ,  sep =  sep_sign)
    df = df.select_dtypes(exclude=['object']) #去掉object类型的数据
    df = df.fillna(df.mean())
    #change to numpy data
    
    data = df.to_numpy()
    data_len = data.shape[0]
    # print(doc_num,word_num)
    train_num = int(data_len /5*4) ; test_num = int(data_len /5)
    index = np.random.permutation(data_len)
    data_train = data[index[: train_num]]
    data_test = data[index[train_num  : ]] 
    x_train = data_train[: , :-1] ; y_train = data_train[: , -1] ; x_test = data_test[: , :-1] ; y_test = data_test[: , -1]
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_train = (x_train - x_mean)/x_std ; x_test =  (x_test - x_mean)/x_std 
    return x_train, y_train, x_test , y_test , x_mean , x_std    #返回训练集和测试集


def SelectFeatures(temp_x , temp_y  , valid_features  , thresh ):  #所有有效的没有被利用的features
    #temp_data: (B , F) ; 
    i_choice = 0 ;  split_value = 0 ; min_MSE = None  ; end_signal = 0
    if np.sum(   (temp_y - np.mean(temp_y))**2 ) < thresh:
        end_signal = 1
        return i_choice ,   split_value ,  min_MSE  , end_signal 
    elif np.all(valid_features == 0):
        end_signal = 1
        return i_choice ,   split_value ,  min_MSE  , end_signal
    elif temp_x.shape[0] == 1:
        end_signal = 1
        return i_choice , split_value , min_MSE , end_signal
    for i in range(temp_x.shape[1]):  #所有的feature
        if valid_features[i] == 0:  #已经选用了的feature
            #print("here i is {}" .format(i))
            continue
        temp_feature = temp_x[: , i] ; sort_index = np.argsort(temp_feature) ; 
        sort_feature = temp_feature[sort_index] ; sort_label = temp_y[sort_index]
        candidate_splits = (sort_feature[:-1] + sort_feature[1:]) / 2  #候选的分割值
        temp_min_MSE = None ; temp_split_value = None
        
        for j , candidate_split in enumerate(candidate_splits):  #choose split number
            temp_label_left = sort_label[:j+1] ; temp_label_right = sort_label[j+1:]
            temp_mean_left = np.mean(temp_label_left) ; temp_mean_right = np.mean(temp_label_right)
            temp_MSE = np.sum((temp_label_left - temp_mean_left)**2) + np.sum((temp_label_right - temp_mean_right)**2)
            if temp_min_MSE is None or temp_MSE < temp_min_MSE:
                temp_min_MSE = temp_MSE ; temp_split_value = candidate_split
            
        if min_MSE is None or temp_min_MSE < min_MSE:
            i_choice = i ; split_value = temp_split_value ; min_MSE = temp_min_MSE
            
    return i_choice ,   split_value ,  min_MSE  , end_signal 
    
def SplitNode(temp_data ,temp_label  , thresh , valid_features ): # if valid_features[i] == 0, then the i-th feature has been used. Otherwise, the i-th feature has not been used.
   #Branch the current node, where samplesUnderThisNode are the samples under the current node, and threshold is the threshold for stopping the branch. Please report the conditions for stopping the branch.
    assert(temp_data.shape[0] == temp_label.shape[0] and temp_data.shape[0] > 0)
    i_choice ,   split_value ,  min_MSE  , end_signal  = SelectFeatures(temp_data ,temp_label , valid_features ,  thresh )
    node_tree = {}
    if end_signal:
        assert(temp_data.shape[0] > 0)
        node_tree["end"] = 1  # end
        node_tree["weight"] = temp_data.shape[0]
        node_tree["c"] = np.mean(temp_label)
        return node_tree
    
    valid_features_new = valid_features.copy()
    #print(valid_features_new[8])
    
    #print("split_value is {} and temp data is {}" .format(split_value , temp_data[: , i_choice]) )
    left_index = np.where(temp_data[:,i_choice] <= split_value)[0]
    right_index = np.where(temp_data[:,i_choice] > split_value)[0]
    temp_data_left = temp_data[left_index , :]
    temp_label_left = temp_label[left_index]
    temp_data_right = temp_data[right_index , :]
    temp_label_right = temp_label[right_index]
    
    if temp_data_left.shape[0] == 0:
        temp_data_left = temp_data[:1 , :]
        temp_label_left = temp_label[:1]
    if temp_data_right.shape[0] == 0:
        temp_data_right = temp_data[:1 , :]
        temp_label_right = temp_label[:1]
    
    valid_features_new[i_choice] = 0
    #print("i_choice is {} ".format(i_choice ))
    #print("valid_features_new ichoice  is {} ".format(valid_features_new[i_choice] )
    node_tree["left"]   =  SplitNode(temp_data_left , temp_label_left , thresh , valid_features_new) 
    node_tree["right"]  = SplitNode(temp_data_right , temp_label_right , thresh , valid_features_new)
    node_tree["feature"] = i_choice
    node_tree["split_value"] = split_value
    node_tree["weight"] = temp_data.shape[0]
    node_tree["c"] = np.mean(temp_label)
    node_tree["end"] = 0  # not end yet
    
    return node_tree

def prune_tree(tree_dict, data_test , label_test):
    if tree_dict["end"] :
        all_mse = np.sum( (label_test - tree_dict["c"])**2 )
        return all_mse
    if data_test.shape[0] == 0:
        return 0. #no valid data
    
    i_choice = tree_dict["feature"] ; split_value = tree_dict["split_value"]
    left_index = np.where(data_test[:,i_choice] <= split_value)[0]
    right_index = np.where(data_test[:,i_choice] > split_value)[0]
    temp_data_left = data_test[left_index , :]
    temp_label_left = label_test[left_index]
    temp_data_right = data_test[right_index , :]
    temp_label_right = label_test[right_index]
    
    tree_left = tree_dict["left"] ; tree_right = tree_dict["right"]
    mse_left = prune_tree(tree_dict["left"], temp_data_left , temp_label_left)
    mse_right = prune_tree(tree_dict["right"], temp_data_right , temp_label_right)

    # 计算当前节点剪枝前后的性能
    error_before_pruning = mse_left + mse_right
    mean_value = (tree_left["c"]*tree_left["weight"] +  tree_right["c"]*tree_right["weight"] )/(tree_left["weight"] + tree_right["weight"])
    error_after_pruning = np.sum((label_test - mean_value)**2)

    # 如果剪枝后性能不下降，则剪枝
    if error_after_pruning <= error_before_pruning:
        tree_dict["left"] = None ; tree_dict["right"] = None ;  tree_dict["end"] = 1
        cur_mse = error_after_pruning
    else:
        cur_mse = error_before_pruning
    return cur_mse


def GenerateTree(data , label , threshold):
    
    thresh = threshold
    valid_features = np.ones(data.shape[1] , dtype = np.int32)
    tree_dict = SplitNode(data , label , thresh , valid_features )
    return tree_dict

        
def Dicision(tree_dict , data_test ):  
    
    category_dict = tree_dict
    
    while not category_dict["end"]:
        if data_test[category_dict["feature"]] <= category_dict["split_value"]:
            category_dict = category_dict["left"]
        else:
            category_dict = category_dict["right"]
    
    assert(category_dict["end"] == 1  )
    return category_dict["c"] #1-9
    




def main(file_path  , threshold):
    RMSE_all = [] ; MAE_all = [] ; R2_all = []
    for k in range(5):
        x_train, y_train, x_test , y_test , x_mean , x_std  = getdata(file_path)
        #print(x_train.shape , y_train.shape  )
        #threshold = 5
        tree_dict = GenerateTree(x_train , y_train , threshold)
        prune_tree(tree_dict, x_test , y_test)
        all_mse = 0 ; all_ae = 0
        for i , x in enumerate(x_test):
            predict_result = Dicision(tree_dict , x)
            #print("predict result is {} and true result is {}" .format(predict_result , y_test[i]))
            all_mse += (predict_result - y_test[i])**2
            all_ae += np.abs(predict_result - y_test[i])
        ss_tot = np.sum((y_test - np.mean(y_test))**2)
        R2 = 1 - (all_mse/ss_tot)
        RMSE = np.sqrt(all_mse/x_test.shape[0]) ; MAE = all_ae/x_test.shape[0]
        #print("expriment is {} RMSE is {} and MAE is {} R2 is {}".format(k , RMSE,MAE , R2))
        RMSE_all.append(RMSE) ; MAE_all.append(MAE) ; R2_all.append(R2)
    
    print("data path is {} \n  threshold is {:.2f} RMSE is {:.2f} and MAE is {:.2f} R2 is {:.2f}".format(file_path ,threshold ,  np.mean(RMSE_all),np.mean(MAE_all) , np.mean(R2_all)))
    

if __name__ == "__main__":
    np.random.seed(0)
    filepaths = [ "./data/HousingData.csv" ,  "./data/wine+quality/winequality-red.csv",  "./data/wine+quality/winequality-white.csv" , "./data/online+news+popularity/OnlineNewsPopularity/OnlineNewsPopularity.csv" ]
    #filepaths = [ "./data/HousingData.csv" ,  "./data/wine+quality/winequality-red.csv" ]
    thresh_all = [0.5 , 1,5,10,50,100]
    for file_path in filepaths:
        for threshold in thresh_all:
            main(file_path , threshold) 
    