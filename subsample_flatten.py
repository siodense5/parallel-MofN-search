import numpy as np

''' subsample_flatten_arrays:

    Input: A set of input arrays with dimensions [num_examples,height,width,num_channels], width, height,
    number of input channels, and center position.
    Output: A 2d array representing the subsampled inputs centered at new_position. 
    
    arrs is the set of input arrays, w and h are the desired width and height of the output arrays given in the
    form w=[w_left,w_right], h=[h_down,h_up] and new_position is center of the desired sub array.
    new_position is given as an integer and used to calculate the  center position of the desired array
    (x_center,y_center). 
    
    For each input channel the method selects the subarray
    [x_center-w_left:x_center+w_right,y_center-h_down,y_center+h_up](using SAME padding) and flattens the collection of
    subarrayrs into a vector of length=(w_right-w_left)*(h_up-h_down)*num_channels.
    
    This is repeated for each input array resulting in an output array of shape [length,num_examples]. 
'''


def subsample_flatten_arrays(arrs,w,h,channels,new_position):
    num_arrs=arrs.shape[0]
    height=arrs.shape[1]
    width=arrs.shape[2]
    out_arrs=[]
    temp_out_arr=[]
    
    subsample_width=w[0]+w[1]+1
    subsample_height=h[0]+h[1]+1
    
    w_pad=0
    h_pad=0
    
    print("Width/Height",width,height)
    
    #Calculate the position of the subarray and any necessary padding
    if new_position>=0:
    
        h_pos=new_position%width
        v_pos=int(new_position/height)       
    
        rw_pad=(w[1]-(width-1-h_pos))*((width-1-h_pos)<w[1])
        dh_pad=(h[1]-(height-1-v_pos))*((height-1-v_pos)<h[1])
        
        lw_pad=(w[0]-h_pos)*(h_pos<w[0])
        uh_pad=(h[0]-v_pos)*(v_pos<h[0])
        
        if (h_pos<w[0]):
            start_h_pos=0
        else:

            start_h_pos=h_pos-w[0]
            
        if (v_pos<h[0]):
            start_v_pos=0
        else:
            start_v_pos=v_pos-h[0]
        
        end_h_pos=h_pos+w[1]+1
        end_v_pos=v_pos+h[1]+1
                
        print("Position:",h_pos,v_pos)
        print("Start/end Horizontal", start_h_pos,end_h_pos)
        print("Start/end verticle", start_v_pos,end_v_pos)
        print("padding: Left, right, up, down", lw_pad,rw_pad,uh_pad,dh_pad)
        
    #Select the given subarray, pad (if necessary), then flatten it. Do this for each feature and concatinate the results
    #for a single example. Repeat for every example
    for arr in arrs:
        temp_out_arr=[]
                
        for i in range(channels):
                
            to_add=np.pad(arr[start_v_pos:end_v_pos,start_h_pos:end_h_pos,i],((uh_pad,dh_pad),(lw_pad,rw_pad)),'constant')
                                
            temp_out_arr.append(np.reshape(to_add,subsample_width*subsample_height))
                
        out_arrs.append(np.reshape(np.array(temp_out_arr),channels*subsample_width*subsample_height))

    return np.array(out_arrs)  


''' Methods to correctly flatten/unflatten arrays representing the weights of a matrix and the activities of
    a hidden/visible layer.
    
    flatten_filters -- transforms a 3d array of weights into a 2d array 
    
    flatten_inputs -- transforms a 4d array of activations of the form 
    [num_examples,height,width,num_channels] into a 2d array of form [num_examples,height*width*num_channels]
    
    unflatten_inputs -- the inverse of flatten_inputs
    
    designed so that unflatten_inputs ( flatten_filters(w) * flatten_inputs(x)) is equivalent to calculating the 4d output
    array of a keras convolutional layer with weights w and input x
'''

def unflatten_inputs(inputs,dimensions):

    num_examples=dimensions[0]
    dim1=dimensions[1]
    dim2=dimensions[2]
    num_channels=dimensions[3]

    unflat_inputs=np.zeros(dimensions)

    for i in range(num_channels):

        unflat_inputs[:,:,:,i]=np.reshape(inputs[:,i*dim1*dim2:i*dim1*dim2+dim1*dim2],[-1,dim1,dim2])

    return unflat_inputs


def flatten_filters(filters):

    num_filters=filters.shape[2]
    num_channels=filters.shape[1]
    weight_size=filters.shape[0]

    weights=np.zeros([weight_size*num_channels,num_filters])

    for i in range(num_filters):
        for j in range(num_channels):
            weights[j*weight_size:j*weight_size+weight_size,i]=filters[:,j,i]

    return weights

def flatten_inputs(inputs):

    num_examples=inputs.shape[0]
    dim1=inputs.shape[1]
    dim2=inputs.shape[2]
    num_channels=inputs.shape[3]

    flat_inputs=np.zeros([num_examples, dim1*dim2*num_channels])

    for i in range(num_channels):

        flat_inputs[:,i*dim1*dim2:i*dim1*dim2 + dim1*dim2]=np.reshape(inputs[:,:,:,i],[-1,dim1*dim2])

    return flat_inputs