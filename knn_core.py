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
    
    #calculate accuracy
    def accuracy(self,df):
        #set class_pred to string
        df['Class_pred'] = df['Class_pred'].astype(str)
        #set class to string
        df['Class'] = df['Class'].astype(str)
        #print(f"Accuracy: {sum(df['Class'] == df['Class_pred'])}/{len(df)}")
        return sum(df['Class'] == df['Class_pred'])/len(df)
    
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

    

    

    