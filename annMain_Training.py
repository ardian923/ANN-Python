# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program aims to train ANN dataset and save weight for later use

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ann_rd_csvData as rd
from scipy import stats

eta   = 0.9#0.9  # Learning rate, 
alpha = 0.1#0.9 # Momentum, 
temp  = 1.0  # Gain, 

# Read ANN data from csv file
nameFile= 'viscTrainData'#'color_aTrain' #filename without csv extension
filename = nameFile+'.csv' #'color_aTrainData.csv' #input ANN data file
rd.read_csv(filename)
Xori = rd.X
Yori = rd.Y

# name of file to save output and weight file
output_filename = nameFile+'_output.csv'
weight_filename = nameFile+'_weight.csv'

# Define input variables
Iteration = 5000
ni = rd.ni
no = rd.no
ncase = rd.ncase

# Normalize Input and Output before training
Xmin = Xori.min(axis=0)
Xmax = Xori.max(axis=0)
Ymin = Yori.min(axis=0)
Ymax = Yori.max(axis=0)
X = (Xori - Xmin)/(Xmax - Xmin)
Y = (0.6*(Yori - Ymin)/(Ymax - Ymin)) + 0.2
#Y = ((Y - Ymin)/(Ymax - Ymin))
print (X)
print (Y)

nl = input('Amount of hidden layer : '); nl = int(nl)
nl = nl + 2 # with input and output layer
nn = np.arange(nl)   #amount of node, create numpy array for number of nodes

i = 0
for nodes in nn:         #update number of nodes for each layer
    if i == 0: 
        nn[i] = ni
    elif i == (nl-1): 
        nn[i] = no
    else:
        nn[i] = input("Amount of nodes in layer "+str(i+1)+" (hidden layer) : ")
        #input()
    i = i + 1
print ('ANN Structure : ', nn)

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

np.random.seed(1)
# randomly initialize our weights with mean 0
synapse = ['syn'+str(i) for i in np.arange(nl-1)]
syn_dict = {}		# set data dictionary for synapse
print ('Synapses : ', synapse)
i = 0
for syn in synapse:
    syn = 2 * np.random.random((nn[i],nn[i+1])) - 1
    syn_dict[synapse[i]]=syn # update data dictionary
    i = i + 1

print (syn_dict)

# set "layer * weight" variable, firt layer = input, end layer = output
layer = ['l'+str(i) for i in np.arange(nl)]
l_dict = {} # set data dictionary for l
print ('Layer : ', layer)

# set synapse's error variable
l_error = ['lerror'+str(i) for i in np.arange(nl)]
l_error_dict = {} # set data dictionary for lerror
print ('Layer Error : ', l_error)

# set synapse's delta variable
l_delta = ['ldelta'+str(i) for i in np.arange(nl)]
l_delta_dict = {} # set data dictionary for ldelta
print ('Layer Delta : ', l_delta)

input("Press Enter to continue...")
    
for j in range(Iteration):
	# Feed forward through layers
    i = 0
    for l in range(nl-1):
        if l == 0: l_dict[layer[0]] = X
        l_dict[layer[i+1]] = nonlin(np.dot(l_dict[layer[i]], syn_dict[synapse[i]]))
        i = i + 1
    # Backpropagation starts here
    for l in layer[::-1]: #looping turun
        # calculate error of model and the target value?
        if i == (nl-1): # calculate error from output
            l_error_dict[l_error[i]] = Y - l_dict[layer[i]] 
            l_delta_dict[l_delta[i]] = l_error_dict[l_error[i]] * \
                                      nonlin(l_dict[layer[i]],deriv=True) 
            # print error average every 1000 counts
            if (j% 1000) == 0: 
                print ('Error:' + str(np.mean(np.abs(l_error_dict[l_error[i]]))))
                print ('Output Prediction : ', l_dict[layer[i]])
        else:
            l_error_dict[l_error[i]] = l_delta_dict[l_delta[i+1]].dot(syn_dict[synapse[i]].T) 
            l_delta_dict[l_delta[i]] = l_error_dict[l_error[i]] * \
                                         nonlin(l_dict[layer[i]],deriv=True) 

        # Updating weight  
#        if (i!=0):
#            syn_dict[synapse[i-1]] += l_dict[layer[i-1]].T.dot(l_delta_dict[l_delta[i]])
#        else:
#            print (i)
#            error = np.mean(np.abs(l_error_dict[l_error[i]]))
#            print ('Error:' + str(error))
#        i = i - 1
        #print ('Output Prediction', l_dict[layer[nl-1]])

        if (i!=0):
            syn_dict[synapse[i-1]] += alpha * (l_dict[layer[i-1]]).T.dot(eta*l_delta_dict[l_delta[i]])
        else:
            print (i)
            error = np.mean(np.abs(l_error_dict[l_error[i]]))
            print ('Error:' + str(error))
        i = i - 1

# Denormalize Output after training
Yhas  = l_dict[layer[nl-1]] # ANN output before denormalize
Y_ANN = ((Yhas - 0.2) * (Ymax - Ymin))/0.6 + Ymin # ANN output after denormalize
#print ('Output After Training Before Normalization : ', Yhas)
#print ('Output After Training After normalization : ', Y_ANN)

#print ('Synapeses dictionary', syn_dict)
#print ('Layer dictionary', l_dict)
#print ('Layer error dictionary', l_error_dict)

# Procedure to save weight in all synapses
with open(weight_filename, 'a') as f:
    f.seek(0)                                                       # find firt line
    f.truncate()                                                    # delete all data below
    f.write(str(nl)+'\n')
    i = 0
    for nodes in nn:
        f.write(str(nn[i])+'\n')
        i = i + 1

    for l in range(nl-1):
        print (l)
        f.write('synapses number (from layer) : '+str(l)+'\n')
        print (syn_dict[synapse[l]].shape)
        syn_shape_row = syn_dict[synapse[l]].shape[0]
        f.write(str(syn_shape_row)+'\n')
        syn_shape_col = syn_dict[synapse[l]].shape[1]
        f.write(str(syn_shape_col)+'\n')
        for r in range(syn_shape_row):
            for c in range(syn_shape_col):
                print (syn_dict[synapse[l]][r,c])
                f.write(str(syn_dict[synapse[l]][r,c])+'\n')        # '\n' => write in new line every loop
    f.write('Error : '+str(error)+'\n')
    f.write('Iteration : '+str(Iteration))

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
outData = np.concatenate((case_no, Xori, Yori, Y_ANN), axis=1) # join matrixes to become complete numpy matrix of dataset and prediction

#np.savetxt(output_filename, outData, delimiter=',')          # save in 'numpy' way

case_no.flatten()#;print(case_no) # column matrix to become 1d numpy array
# Create pandas data frame
df = pd.DataFrame(data=outData.astype(float), columns=df_col) # pandas data frame
print ("ANN Training Result Data Frame : ")
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
ax1.set_title('Comparison Data and ANN Prediction')
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
ax2.set_title('Correlation between Data and ANN Prediction')
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






