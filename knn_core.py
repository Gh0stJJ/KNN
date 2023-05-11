import pandas as pd
#class KNN

class KNN:
    def __init__(self):
        pass

    # n dimensional eclidian distance
    def distance(self,x, y):
        return sum([(x[i]-y[i])**2 for i in range(len(x))])**0.5
    
    # get k nearest neighbors
    def knn(self,df_test, df_train, k):
        #for each row in test dataframe
        for index, row in df_test.iterrows():
            #calculate distance between test point and all points in dataframe
            #print('Test point: ', row[:-1])
            df_train['Distance'] = df_train.apply(lambda rowT: self.distance(rowT[:-1], row[:-1]), axis=1)

            #sort by distance
            df_train.sort_values(by=['Distance'], inplace=True)
            #print('Train points: ')
            #print(df_train)
            #get k nearest neighbors
            neighbors = df_train.head(k)
            #get most common class
            #print('Class: ')
            classi = neighbors['Class'].value_counts().idxmax()
            #print(classi)

            #add class_pred column to test dataframe
            df_test.loc[index, 'Class_pred'] = str(classi)

            #drop distance column
            df_train.drop(columns=['Distance'], inplace=True)
            
        return df_test
    
    #get table for each test point

    def knn_df(self,df_test, df_train):

        #dictionary to save the dataset with index
        df_dict = {}

        for index, row in df_test.iterrows():
            #calculate distance between test point and all points in dataframe
            df_train['Distance'] = df_train.apply(lambda rowT: self.distance(rowT[:-1], row[:-1]), axis=1)

            #sort by distance
            df_train.sort_values(by=['Distance'], inplace=True)
            #save the dataset with the test instance
            df_dict[index] = df_train.copy()
            #drop distance column
            df_train.drop(columns=['Distance'], inplace=True)
           
        return df_dict
    
    #calculate accuracy
    def accuracy(self,df):
        #set class_pred to string
        df['Class_pred'] = df['Class_pred'].astype(str)
        #set class to string
        df['Class'] = df['Class'].astype(str)
        print(f"Accuracy: {sum(df['Class'] == df['Class_pred'])}/{len(df)}")
        return sum(df['Class'] == df['Class_pred'])/len(df)
    
    #get k nearest neighbors

    def opt_knn(self, df_test, df_list, k):

        for index, row in df_test.iterrows():
            #get the dataset with the test instance
            df_train = df_list.get(index)
            #get k nearest neighbors
            neighbors = df_train.head(k)
            #get most common class
            classi = neighbors['Class'].value_counts().idxmax()
            #add class_pred column to test dataframe
            df_test.loc[index, 'Class_pred'] = str(classi)

        return df_test
    

    def opt_best_k(self,df_test: pd.DataFrame, df_train: pd.DataFrame) -> tuple:
        #for each k from 1 to 10
        best = 0
        #Best k and accuracy tuple
        best_k_tup = (0,0)
        #get list of dataframes
        df_list = self.knn_df(df_test, df_train)
        for k in range(1, len(df_train)):
            #get predictions
            df_test = self.opt_knn(df_test, df_list, k)
            #calculate accuracy
            acc = self.accuracy(df_test)
            print('k: ', k, 'accuracy: ', acc)
            #if accuracy is better than best
            if acc > best:
                #update best
                best = acc
                best_k_tup = (k, acc)
            #print('k: ', k, 'accuracy: ', acc)

        return best_k_tup
            #

    
    #find best k
    def best_k(self,df_test: pd.DataFrame, df_train: pd.DataFrame) -> tuple:
        #for each k from 1 to 10
        best = 0
        #Best k and accuracy tuple
        best_k_tup = (0,0)
        for k in range(1, 99):
            #get predictions
            df_test = self.knn(df_test, df_train, k)
            #calculate accuracy
            acc = self.accuracy(df_test)
            #if accuracy is better than best
            if acc > best:
                #update best
                best = acc
                best_k_tup = (k, acc)
            #print('k: ', k, 'accuracy: ', acc)

        return best_k_tup
    
    # get class by k

    def get_class(self,df_test: pd.DataFrame, df_total: pd.DataFrame, k: int) -> str:

        #get the fist row of the test dataframe
        row = df_test.iloc[0]
        #calculate distance between test point and all points in dataframe
        df_total['Distance'] = df_total.apply(lambda rowT: self.distance(rowT[:-1],row), axis=1)

        #sort by distance
        df_total.sort_values(by=['Distance'], inplace=True)

        print(df_total)
        #get k nearest neighbors
        neighbors = df_total.head(k)
        #get most common class
        print("-----------------Frecuencias------------------------")
        print(neighbors['Class'].value_counts())
        classi = neighbors['Class'].value_counts().idxmax()
        #drop distance column
        df_total.drop(columns=['Distance'], inplace=True)

        df_test['Class_pred'] = str(classi)
        return df_test

    

    

    