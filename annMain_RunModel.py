# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program aims to run ANN model for a dataset based on weight resulted from ANN training

import sys
import numpy as np
import pandas as pd
import ann_rd_csvData as rd
import ann_rd_csvWeight as rw
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

#Not used here
#eta   = 0.9#0.9  # Learning rate, 
#alpha = 0.1#0.9 # Momentum, 
#temp  = 1.0  # Gain, 

# name of file to read data and save output file
nameFile= 'viscTestData'# filename without csv extension
filename = nameFile+'.csv' #input ANN data file to read
output_filename = nameFile+'_output.csv' # name of file to save output file

# read weight
weight_filename = 'viscTrainData_weight.csv' #put weight file from TRAINING RESULT
rw.read_weight(weight_filename)
w_ni = rw.ni
w_no = rw.no
nl   = rw.nl
synapse  = rw.synapse
syn_dict = rw.syn_dict

# read data
rd.read_csv(filename)
Xori = rd.X
Yori = rd.Y
ni = rd.ni
no = rd.no
ncase = rd.ncase

# Normalize Input and Output before running model
Xmin = Xori.min(axis=0)
Xmax = Xori.max(axis=0)
Ymin = Yori.min(axis=0)
Ymax = Yori.max(axis=0)
X = (Xori - Xmin)/(Xmax - Xmin)
Y = (0.6*(Yori - Ymin)/(Ymax - Ymin)) + 0.2
print (X)
print (Y)

# evaluate weight and data, are them equal input and output?
if (w_ni != ni) or (w_no != no):
    sys.exit("input or otuput variabel in dataset and weight file is not equal")

input("Press Enter to continue...")

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

# set "layer * weight" variable, firt layer = input, end layer = output
layer = ['l'+str(i) for i in np.arange(nl)]
l_dict = {} # set data dictionary for l
#print ('Layer : ', layer)

# Feed forward through layers
i = 0
for l in range(nl-1):
    if l == 0: l_dict[layer[0]] = X
    l_dict[layer[i+1]] = nonlin(np.dot(l_dict[layer[i]], syn_dict[synapse[i]]))
    i = i + 1
Y_ANN = l_dict[layer[nl-1]]    

# Denormalize Output after run model
Y_ANN = ((Y_ANN - 0.2) * (Ymax - Ymin))/0.6 + Ymin
#print (Y_ANN)

# create output file that compare output data and ANN output side by side
with open(output_filename, 'a') as f:
    f.seek(0)                                                       # find first line
    f.truncate()                                                    # delete all data below
    # first three lines is header file
    f.write('ni,no,ncase'+'\n')
    f.write(str(ni)+','+str(no)+','+str(ncase)+'\n')

# writing second column header for output file
ncase_ = ['case_no']
x_in = ['xinp'+str(i+1) for i in np.arange(ni)]
y_out= ['yout'+str(i+1) for i in np.arange(no)]
y_ANN= ['y_ANN'+str(i+1) for i in np.arange(no)]
df_col = np.concatenate((ncase_, x_in, y_out, y_ANN), axis=0) # join matrixes for data frame columns (column name)

case_no = np.arange(ncase+1); case_no = case_no[1:] # slice first column
case_no = np.array(case_no)[np.newaxis] # create row matrix
case_no = case_no.T # create column matrix for case no
# print (case_no)
outData = np.concatenate((case_no, Xori, Yori, Y_ANN), axis=1) # join matrixes to become complete numpy matrix of dataset and prediction

#print ('output ANN : ', outData)
#np.savetxt(output_filename, outData, delimiter=',')          # save in 'numpy' way

case_no.flatten()#;print(case_no) # column matrix to become 1d numpy array
# Create pandas data frame
df = pd.DataFrame(data=outData.astype(float), columns=df_col) # pandas data frame
print ("ANN Prediction Result Data Frame : ")
print (df)
with open(output_filename, 'a') as f:
    df.to_csv(f, sep=',', header=True, float_format='%.9f', index=False) # save in 'pandas' way

# Calculate errors and trendline (simply a linear fitting)
RMSE = ((df.y_ANN1 - df.yout1) ** 2).mean() ** .5
print ('RMSE : ', RMSE)
slope, intercept, r_value, p_value, std_err = stats.linregress(Yori.flatten(), Y_ANN.flatten())
print ('slope ', slope)
print ('intercept ', intercept)
print ('r value', r_value)
print ('p_value', p_value)
print ('standard deviation', std_err)
line = slope*(Yori)+intercept

# Plot Data vs ANNOutput (time series) and Data vs ANNOutput (Correlation)
mpl.rc('font', family='serif') 
mpl.rc('font', serif='Helvetica Neue')
#mpl.rc('text', usetex='true')
mpl.rcParams.update({'font.size': 14})

#plt.figure(1)
#fig = plt.figure()
#ax1 = fig.add_subplot(131)
fig, ax1 = plt.subplots()
ax1.plot(case_no, Yori, 'ro', label='Data')
ax1.plot(case_no, Y_ANN, 'b-', label='ANN Prediction')
ax1.set_title(nameFile)
ax1.legend(loc='lower right', ncol=1, shadow=False, frameon=False)
ax1.set_xlabel('Case No.')
ax1.set_ylabel('Output')
#ax1.set_xlim(xmax = 0.45, xmin = 0.15)
#ax1.set_ylim(ymax = 0.45, ymin = 0.15)
fig.savefig(nameFile+'_Figure1.png')

####

#ax2 = fig.add_subplot(132)
fig, ax2 = plt.subplots()
ax2.plot(Yori, Y_ANN, 'ro')
ax2.plot(Yori, line, 'b-', label='Correlation')
ax2.set_title('Correlation : '+nameFile)
ax2.legend(loc='upper left', shadow=False, frameon=False)
ax2.set_xlabel('Output Data')
ax2.set_ylabel('Output ANN Prediction')
#set location for regrssion equation and r
xx = Yori.min(axis=0) + 0.40 * (Yori.max(axis=0) - Yori.min(axis=0))
yy = Y_ANN.min(axis=0) + 0.15 * (Y_ANN.max(axis=0) - Y_ANN.min(axis=0))  
yy2= yy - 0.14 * (Y_ANN.max(axis=0) - Y_ANN.min(axis=0))
ax2.text(xx, yy, '$ANN Pred. = $'+str('{:.2f}'.format(slope))+' $* Data +$ '+str('{:.2f}'.format(intercept)))
ax2.text(xx, yy2, '$r = $'+str('{:.2f}'.format(r_value)))
#ax2.set_xlim(xmax = 1.0, xmin = 0.0)
#ax2.set_ylim(ymax = 1.0, ymin = 0.0)
fig.savefig(nameFile+'_Figure2.png')

plt.show()

