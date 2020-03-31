# parallel-MofN-search
Decompositional rule extraction method for neural networks. Binarizes continuous activation values using information gain and searches through a set of MofN rules relating input and output neurons in a layer. Uses a simple loss function weighing the complexity and accuracy of the rules to choose the optimal rule to each output neuron

For more details on the algorithm see https://kr2ml.github.io/2019/papers/KR2ML_2019_paper_16.pdf

TO USE: open parallel_MofN_search.ipynb, enter the necessary preamble to setup spark using the cloud service of your choice in the first cell. Run the cells one by one to extract rule from the example network

To use on your own data replace the paths in the second cell with the paths to the data you wish to test. In the third cell, if you are extracting from the final layer of a network set Final_Layer=True and if you are extracting from a convolutional layer set subsample=True and set num_input_channels to the number of channels in the input and feature_dimension to the size of the convolutional window of the layer you are extracting from.

If you wish to only extract rules for a selection of hidden units change start_rule and end_rule in the 4th cell.

FORMATTING OF INPUT: To run the algorithm on layer L of a network it must be provided with 1)The layer weights, 2)activations of layer L-1 on some set of test examples, 3)the corresponding activations of layer L, 4)The activations of the output layer on the test set

Each of these must be in the form of a 2-dimensional array stored in a csv file. For convonlutional data, use the methods in subsample_ flatten to convert weights and activations into 2-dimensional arrays. The output layer given to the algorithm should either be activation values or logits rather than output labels. The activation values are used to convert the output layer to either integer labels or a one-hot encoding

Originally implemented using ibm cloud services with Watson studio. If you have difficulty getting it to run on your cloud platform, have any questions, or see areas with possible improvements please contact me at simon.odense@city.ac.uk
