# modified from http://iamtrask.github.io/2015/07/12/basic-python-network/
# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program aims to train ANN dataset and save weight for later use

import numpy as np
import rd_csvANN_Data as rd

# Learning rate
# Momentum
# Gain

# Read ANN data from csv file
filename = 'ORGate_data.csv' #input ANN data file
rd.read_csv(filename)
X = rd.X
Y = rd.Y

# Name of file to save weight file
weight_filename = 'weight_.csv'

# Define input variables
Iteration = 10000
ni = rd.ni
no = rd.no
ncase = rd.ncase

nl = input('Amount of hidden layer : ')
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
        #raw_input()
        i = i + 1
print 'ANN Structure : ', nn

def nonlin(x,deriv=False):
	if(deriv==True):
	    return x*(1-x)

	return 1/(1+np.exp(-x))

np.random.seed(1)
# randomly initialize our weights with mean 0
synapse = ['syn'+str(i) for i in np.arange(nl-1)]
syn_dict = {}		# set data dictionary for synapse
print 'Synapses : ', synapse
i = 0
for syn in synapse:
	syn = 2 * np.random.random((nn[i],nn[i+1])) - 1
	syn_dict[synapse[i]]=syn # update data dictionary
	i = i + 1

# set "layer * weight" variable, firt layer = input, end layer = output
layer = ['l'+str(i) for i in np.arange(nl)]
l_dict = {} # set data dictionary for l
print 'Layer : ', layer

# set synapse's error variable
l_error = ['lerror'+str(i) for i in np.arange(nl)]
l_error_dict = {} # set data dictionary for lerror
print 'Layer Error : ', l_error

# set synapse's delta variable
l_delta = ['ldelta'+str(i) for i in np.arange(nl)]
l_delta_dict = {} # set data dictionary for ldelta
print 'Layer Delta', l_delta

raw_input("Press Enter to continue...")
    
for j in xrange(Iteration):
	# Feed forward through layers
    i = 0
    for l in xrange(nl-1):
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
                print 'Error:' + str(np.mean(np.abs(l_error_dict[l_error[i]])))
                print 'Output Prediction : ', l_dict[layer[i]]
        else:
            l_error_dict[l_error[i]] = l_delta_dict[l_delta[i+1]].dot(syn_dict[synapse[i]].T) 
            l_delta_dict[l_delta[i]] = l_error_dict[l_error[i]] * \
                                         nonlin(l_dict[layer[i]],deriv=True) 

        # Updating weight  
        if (i<>0):
            syn_dict[synapse[i-1]] += l_dict[layer[i-1]].T.dot(l_delta_dict[l_delta[i]])
        else:
            print i
            error = np.mean(np.abs(l_error_dict[l_error[i]]))
            print 'Error:' + str(error)
        i = i - 1
        print 'Output Prediction', l_dict[layer[nl-1]]

#print 'Synapeses dictionary', syn_dict
#print 'Layer dictionary', l_dict
#print 'Layer error dictionary', l_error_dict

# Procedure to save weight in all synapses
with open(weight_filename, 'a') as f:
    f.seek(0)                                                       # find firt line
    f.truncate()                                                    # delete all data below
    f.write(str(nl)+'\n')
    i = 0
    for nodes in nn:
        f.write(str(nn[i])+'\n')
        i = i + 1

    for l in xrange(nl-1):
        print l
        f.write('synapses number (from layer) : '+str(l)+'\n')
        print syn_dict[synapse[l]].shape
        syn_shape_row = syn_dict[synapse[l]].shape[0]
        f.write(str(syn_shape_row)+'\n')
        syn_shape_col = syn_dict[synapse[l]].shape[1]
        f.write(str(syn_shape_col)+'\n')
        for r in xrange(syn_shape_row):
            for c in xrange(syn_shape_col):
                print syn_dict[synapse[l]][r,c]
                f.write(str(syn_dict[synapse[l]][r,c])+'\n')        # '\n' => write in new line every loop
    f.write('Error : '+str(error)+'\n')
    f.write('Iteration : '+str(Iteration))
