import numpy as np
import rd_csvANN_Data as rd   #read csv files, and make column header as variable name

filename = 'ORGate_data.csv' #input ANN data file
rd.read_csv(filename)

print rd.columnHeaders1
print rd.data_struct
print rd.columnHeaders2
print "X = ", rd.X
print "Y = ", rd.Y
