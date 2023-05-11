import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from knn_core import KNN
import tkinter as tk
from tkinter.filedialog import askopenfilename
import os

#Main menu
root = tk.Tk()
root.withdraw()

def show_dataframe(df):
    root = tk.Tk()
    root.title("DataFrame Viewer")
    text = tk.Text(root)
    text.pack()
    # Agrega el DataFrame al widget de texto
    text.insert(tk.END, df.to_string())
    # Inicia el bucle de eventos
    root.mainloop()
    
if __name__ == "__main__":
    
    while True:
        # Clearing the Screen
        os.system('cls')
        print("Menu K-Nearst Neighbors")
        print("Elige una opcion: ")
        print("1. Cargar datos")
        print("2. Graficar datos")
        print("3. Encotrar mejor K")
        print("4. Clasificar nuevo dato")
        print("5. Salir")
        option = input(": ")

        if option == "1":
           #Show file dialog
            filename = askopenfilename(title="Select file")
            root.update()
            root.destroy()
           
            if filename:
                df = pd.read_csv(filename, header=None)
                print(f"Data loaded {df.shape[0]} rows and {df.shape[1]} columns")
                col_count = df.shape[1]
                #print dataframe in a window
                #add column names
                for i in range(col_count-1):
                    df.rename(columns={i: 'X'+str(i)}, inplace=True)
                df.rename(columns={col_count-1: 'Class'}, inplace=True)
                show_dataframe(df)
                #Separate data into train and test
                train = df.loc[:int(df.shape[0]*0.8)-1, :].copy()
                # tomar el 20% de los datos para prueba sin aleatoriedad
                test = df.loc[int(df.shape[0]*0.8):, :].copy()
                # actualizar los valores del DataFrame original usando .loc[]
                train.loc[:, :] = df.loc[:int(df.shape[0]*0.8), :]
                test.loc[:, :] = df.loc[int(df.shape[0]*0.8):, :]
                input("Presione una tecla para continuar...")
                
            else:
                print("No file selected")
        elif option == "2":
            print("Graficando...")
            if col_count == 3:
                sns.scatterplot(x='X0', y='X1', hue='Class', data=df)
                plt.show()
            elif col_count == 4:
                #plot scatterplot in 3d
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(df['X0'], df['X1'], df['X2'], c=df['Class'], marker='o')
                ax.set_xlabel('X0')
                ax.set_ylabel('X1')
                ax.set_zlabel('X2')
                plt.show()
            else:
                print('Cannot plot data with more than 3 dimensions')
        elif option == "3":
            print("Encontrar el mejor K")
            #knn class
            knn_core = KNN()
            #train knn
            k_opt = knn_core.opt_best_k(test, train)
            print("--------------------------------------------------------")
            print(f"Best K: {k_opt[0]} with {k_opt[1]*100}% accuracy")
            input("Presione una tecla para continuar...")
            
        elif option == "4":
            print("Clasificacion de nuevo dato segun K")
            #new instance
            knn_core = KNN()
            new_instance = []
            for i in range(col_count-1):
                new_instance.append(float(input(f"X{i}: ")))

            k_val = int(input("K: ? "))
            
            #list to dataframe
            new_instance = pd.DataFrame([new_instance])
            #give column names
            for i in range(col_count-1):
                new_instance.rename(columns={i: 'X'+str(i)}, inplace=True)
            
            #knn class
            print(knn_core.get_class(new_instance, train, k_val))
            input("Presione una tecla para continuar...")
                
            
        elif option == "5":
            print("Saliendo...")
            break
            
        else:
            print("Invalid option")
        option = ""
    print("Exiting program...")



