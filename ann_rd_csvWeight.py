# written by : Dr. Ardiansyah, Bio_Environmental Management and Control Engineering, Jenderal Soedirman University (ardi.plj@gmail.com)
# this program contain function that read ANN weight data

import numpy as np

def read_weight(weight_filename):
    with open(weight_filename, 'r') as f:
        #f.readline() # skip the first line (contain information)
        # read ANN structure, number of layer
        global nl
        nl = int(f.readline())                    # Read first line of csv
        nn = np.arange(nl)   #amount of node, create numpy array for number of nodes in each layer

        # read number of nodes in each layer
        i = 0
        for nodes in nn:                     #update number of nodes for each layer 
            nn[i] = int(f.readline())
            i = i + 1
        global ni, no
        ni = nn[0]
        no = nn[nl - 1]

        # create data dictionary and its content
        global synapse     # an array contain dictionary key for synapses
        synapse = ['syn'+str(i) for i in np.arange(nl-1)]
        global syn_dict
        syn_dict = {}		                 # set data dictionary for synapse
        i = 0
        for syn in synapse:
	        syn = 2 * np.random.random((nn[i],nn[i+1])) - 1
	        syn_dict[synapse[i]]=syn         # update data dictionary
	        i = i + 1
        print (syn_dict)

        # read all weight in synapses
        for l in range(nl-1):
            print (l)
            f.readline()                     # skip line contain synapse number
            syn_shape_row = int(f.readline())
            syn_shape_col = int(f.readline())
#            synapse[l] = np.arange(syn_shape_row * syn_shape_col).reshape(syn_shape_row, syn_shape_col)
            for r in range(syn_shape_row):
                for c in range(syn_shape_col):
                    syn_dict[synapse[l]][r,c] = float(f.readline())
    print ('Synapses dictionary data : ', syn_dict)
    return
