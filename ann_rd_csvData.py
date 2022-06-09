# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program contain function that read ANN data, example of data (case of OR Gate)
# ========================================
# ni,no,ncase
# 2,1,4
# case_no,xinp1,xinp2,yout1
# 1,1,1,1
# 2,1,0,1
# 3,0,1,1
# 4,0,0,0
# ========================================
import numpy as np

def read_csv(filename):
    # Read column headers (to be variable naames) and ANN datas
    with open(filename) as f:
        #Read column headers of data structure
        firstline = f.readline()                    # Read first line of csv
        firstline = firstline.replace("\n","")      # Remove new line characters
        firstline = firstline.replace(" ","")       # Remove spaces
        global columnHeaders1
        columnHeaders1 = firstline.split(",")       # Get array of column headers
        print ('Column Header 1 : ', columnHeaders1)
        #Rad data structure of ANN
        secondline = f.readline()                   # Read second line of csv
        secondline = secondline.replace("\n","")    # Remove new line characters
        secondline = secondline.replace(" ","")     # Remove spaces
        global data_struct
        data_struct = [int(val) for val in secondline.split(",")]
        print ('Data Structure : ', data_struct)
        #Read column headers of ANN datas 
        thirdline = f.readline()                    # Read third line of csv
        thirdline = thirdline.replace("\n","")      # Remove new line characters
        thirdline = thirdline.replace(" ","")       # Remove spaces    
        global columnHeaders2
        columnHeaders2 = thirdline.split(",")       # Get array of column headers #array of data variable name
        print ('Column Header 2 : ', columnHeaders2)
        # Read all ANN datas (omitting the three row containing column headers and data information)
        data=np.loadtxt(filename,skiprows=3,delimiter=",",ndmin=2)  #load all ANN data in file
        print ('All Data  : ', data)

    # Assign the data_struct to arrays, with names of the variables generated from column headers
    Ind=0
    for Var in columnHeaders1:
        globals()[Var]=data_struct[Ind]   # Assign the columns of the data to variables names after the column headers
        Ind=Ind+1

    # Assign the data to arrays, with names of the variables generated from column headers              
    Ind=0
    for Var in columnHeaders2:
        globals()[Var]=data[:,Ind]        # Assign the columns of the data to variables names after the column headers
        Ind=Ind+1

    # Exclude case_no from data (slicing matrix)
    idx_IN_columns = [i+1 for i in np.arange(ni+no)]
    extractedData = data[:,idx_IN_columns] # slice columns after first column
    
    idx_IN_columnsX = [i for i in np.arange(ni)]
    idx_IN_columnsY = [i+ni for i in np.arange(no)]
    
    global X, Y
    X = extractedData[:,idx_IN_columnsX]
    Y = extractedData[:,idx_IN_columnsY]
    print ("Input Data (X) = ", X)
    print ("Output Data (Y) = ", Y)
    
    return
    
