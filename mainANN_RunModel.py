# modified from http://iamtrask.github.io/2015/07/12/basic-python-network/
# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program aims to run ANN model for a dataset based on weight resulted from ANN training

import sys
import numpy as np
import pandas as pd
import rd_csvANN_Data as rd
import rd_csvANN_Weight as rw

# Learning rate
# Momentum
# Gain

# name of file to save output file
output_filename = 'outputANN.csv'

# read weight
weight_filename = 'weight_.csv'
rw.read_weight(weight_filename)
w_ni = rw.ni
w_no = rw.no
nl   = rw.nl
synapse  = rw.synapse
syn_dict = rw.syn_dict

# read data
filename = 'ORGate_data.csv' #input ANN data file
rd.read_csv(filename)
X = rd.X
Y = rd.Y
ni = rd.ni
no = rd.no
ncase = rd.ncase

# evaluate weight and data, are them equal input and output?
if (w_ni <> ni) or (w_no <> no):
    sys.exit("input or otuput variabel in dataset and weight file is not equal")

raw_input("Press Enter to continue...")

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

# set "layer * weight" variable, firt layer = input, end layer = output
layer = ['l'+str(i) for i in np.arange(nl)]
l_dict = {} # set data dictionary for l
#print 'Layer : ', layer

# Feed forward through layers
i = 0
for l in xrange(nl-1):
    if l == 0: l_dict[layer[0]] = X
    l_dict[layer[i+1]] = nonlin(np.dot(l_dict[layer[i]], syn_dict[synapse[i]]))
    i = i + 1
Y_ANN = l_dict[layer[nl-1]]    
print Y_ANN

# create output file that compare output data and ANN output side by side
with open(output_filename, 'a') as f:
    f.seek(0)                                                       # find first line
    f.truncate()                                                    # delete all data below
    # first three lines is header file
    f.write('ni,no,ncase'+'\n')
    f.write(str(ni)+','+str(no)+','+str(ncase)+'\n')
    # writing second column header
#    f.write('case_no'+',')
#    i = 0
#    for i in xrange(ni):
#        f.write('xinp'+str(i+1)+',')
#        i = i + 1
#    i = 0
#    for i in xrange(no):
#        f.write('yout'+str(i+1)+',')
#        i = i + 1
#    i = 0
#    for i in xrange(no):
#        if (i+1) == no:
#            f.write('y_ANN'+str(i+1))
#        else:	
#            f.write('y_ANN'+str(i+1)+',')
#        i = i + 1   

ncase_ = ['case_no']
x_in = ['xinp'+str(i+1) for i in np.arange(ni)]
y_out= ['yout'+str(i+1) for i in np.arange(no)]
y_ANN= ['y_ANN'+str(i+1) for i in np.arange(no)]
df_col = np.concatenate((ncase_, x_in, y_out, y_ANN), axis=0) # join matrixes for data frame columns (column name)

case_no = np.arange(ncase+1)
case_no = case_no[1:] # slice first column
case_no = np.array(case_no)[np.newaxis]
case_no = case_no.T
print case_no
outData = np.concatenate((case_no,X, Y, Y_ANN), axis=1) # join matrixes
print 'output ANN : ', outData
#np.savetxt(output_filename, outData, delimiter=',')          # save in 'numpy' way
df = pd.DataFrame(data=outData.astype(float), index=case_no, columns=df_col) # pandas data frame
print "Data Frame : "
print df
with open(output_filename, 'a') as f:
    df.to_csv(f, sep=',', header=True, float_format='%.9f', index=False) # save in 'pandas' way


