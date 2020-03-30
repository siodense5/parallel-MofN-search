import numpy as np

''' generate_splits_parallel: 
    Input: array of activation values from the input layer, integer array of output classes, the number of classes
    Output: array of size (number of features) * 2 representing the best splitting values of each feature and their
            associated information gains

    Generate splits for the features in inputs based on the information gain of the output labels. This is done with a brute 
    force search by listing all values that a feature takes in the array inputs and calculating the information
    gain associated with splitting the feature into a binary variable using that input value.
    The splitting value with the maximum information gain is returned.
    
    the best split for each feature is calculated in parallel and the returned array represents the best splits and
    information gains for every feature
    
    compute_best_split_single_feature:
    submethod of generate_splits_parallel, implements the actual search described above for a single feature
    
'''

def generate_splits_parallel(inputs, output, num_classes):
    
    num_hidden=output.shape[1]
    num_features=inputs.shape[1]
    num_examples=inputs.shape[0]

    ratios=[0]*num_classes
    
    for i in range(num_classes):
        ratios[i]=np.count_nonzero(output==i,axis=0)/num_examples
        
    initial_cross_entropies=np.zeros(num_hidden)
    
    for i in range(num_classes):
        initial_cross_entropies+=-np.nan_to_num(ratios[i]*np.log(ratios[i]))

    rdd1=sc.parallelize([[inputs[:,i],initial_cross_entropies] for i in range(num_features)],500)
    
    rdd2=rdd1.map(lambda x: compute_best_split_single_feature(x[0],x[1],output,num_classes)) 
    
    return np.array(rdd2.collect())

    
def compute_best_split_single_feature(inputs, initial_ce, output,num_classes):
                    
    num_hidden=output.shape[1]
    num_examples=inputs.shape[0]
        
    best_information_gain=np.zeros(num_hidden)
    VALUES=np.unique(inputs)
    best_splits=np.zeros(num_hidden)

    for value in VALUES:
        output_sat=output[np.where(inputs>=value)]
        output_nsat=output[np.where(inputs<value)]

        num_sat=output_sat.shape[0]
        num_nsat=output_nsat.shape[0]
        
        #Calculate the cross entropy of the output labels for which the inputs are greater than or equal to the test
        #splitting value
        sat_ratios=[0]*num_classes

        for i in range(num_classes):
            sat_ratios[i]=np.count_nonzero(output_sat==i,axis=0)/num_sat

        sat_cross_entropy=np.zeros(num_hidden)
        
        for i in range(num_classes):
            sat_cross_entropy+=-np.nan_to_num(sat_ratios[i]*np.log(sat_ratios[i]))
        
        #Calculate the cross entropy for the outputs for which the inputs are less than the test splitting value
        nsat_ratios=[0]*num_classes
        
        for i in range(num_classes):
            nsat_ratios[i]=np.count_nonzero(output_nsat==i,axis=0)/num_nsat

        nsat_cross_entropy=np.zeros(num_hidden)
        
        for i in range(num_classes):
            nsat_cross_entropy+=-np.nan_to_num(nsat_ratios[i]*np.log(nsat_ratios[i]))
        
        #Calculate the difference between the initial entropy of the outputs and the average entropy when the outputs are
        #split by the input feature on the test value to get the information gain for the split on that test value
        
        information_gain=initial_ce- num_sat/num_examples * sat_cross_entropy - num_nsat/num_examples * nsat_cross_entropy

        for j in range(num_hidden):
            if information_gain[j]>best_information_gain[j]:
                best_information_gain[j]=information_gain[j]
                best_splits[j]=value

    return [best_splits,best_information_gain]