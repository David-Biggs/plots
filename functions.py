import statistics
import numpy as np
import os
import statistics as stats
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.metrics import mean_absolute_error as MAE
import math


def get_results(path_2_txt):
    new_list =[]

    with open(path_2_txt, "r") as file:
        lst = file.readlines()

    for i in range(len(lst)):
        new_list.append(float(lst[i].strip("\n")))
        
    return new_list


def get_probs(results_list):
    last = len(results_list)
    probs_list = results_list[4:last:5]
    return probs_list


def get_count(probs_list,threshold):
    predicted_count = np.count_nonzero(np.array(probs_list)>threshold)
    return predicted_count


def get_GT(name):
    name = name.split("!")
    name = str(name[1])
    ground_truth = name.split(".")
    return int(ground_truth[0])


def get_acc(ground_truth,prediction,name,percentage):
    error = 0
    diff = prediction - ground_truth


    if ground_truth > 0: 
        if percentage == True:

                #MAE
                error = abs(prediction - ground_truth)/ground_truth
                #img_acc = ( 1- abs(1 - acc)) 
                #acc = 1 - error
                
        else:
            error = abs(prediction - ground_truth)

    else:
        print("GT: 0 - File name: ",name)
    return error 



def plot_func(x,y,vid_dict):
    plt.rcParams["figure.figsize"] = (20,12) 

    plt.xlabel('x - Threshold')
    plt.ylabel('y - Percentage Error')

    # Plot Prediction
    #plt.plot(x, y, linewidth=3,linestyle='-',label = "Average")

    ymax = min(y)
    xpos = y.index(ymax)
    xmax = x[xpos]
    text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
    #plt.annotate(text,xy=(xmax,ymax),xytext=(xmax,ymax+0.007))

    #plt.plot(xmax, ymax, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")

    if vid_dict:
        for i in vid_dict:
            plt.plot(x, vid_dict[i], linewidth=2, linestyle='--', label = i)
            y = vid_dict[i]
            ymax = min(y)
            xpos = y.index(ymax)
            xmax = x[xpos]
            text = "x={:.3f}, y={:.3f}".format(xmax, ymax)
            #plt.annotate(text,xy=(xmax,ymax),xytext=(xmax,ymax+0.007))

            plt.plot(xmax, ymax, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="red")


    plt.title('Accuracy at various thresholds. Model-HTC')

    # Show legend
    plt.legend()
    # Add grid 
    plt.grid()
    # Show plot
    plt.show()

# Function to plot counts per frame
# def plot_frame_counts(Frame_number,frame_count,ground_truth, vid_name):

#     name = vid_name.split('/')
    
#     import matplotlib.pyplot as plt

#     #   Define the size of the Figure
#     plt.rcParams["figure.figsize"] = (14,8) 

#     # x axis values
#     x = Frame_number
#     plt.xlabel('x - Frame')

#     # corresponding y axis values - counts
#     y = frame_count
#     plt.ylabel('y - Count')


#     # Calculate the total average of counts[]
#     ave_pred = statistics.mean(frame_count)

#     # Plot Prediction
#     plt.plot(x, y, linewidth=1,linestyle='-', label = "Prediction")

#     # Plot Ground Truth - line 
#     plt.axhline(y=ground_truth, xmin=0.05, xmax=max(Frame_number), color='r', linestyle='-.', linewidth=1, label = "Ground Truth")

#     # Plot Prediction Average 
#     plt.axhline(y=ave_pred, xmin=0.05, xmax=max(Frame_number), color='b', linestyle='-.', linewidth=1, label = "Prediction Average")
    
#     # Plot Moving average
#     #plt.plot(x, ma, linewidth=1, linestyle='-.',label = "Prediction Moving Average")

    
#     # Title
#     plt.title('Count over mulitple frames from:'+ name[8])

#     # Show legend
#     plt.legend() 

#     # Show plot
#     plt.show()

#     # Error calculations
#     error_list = []
#     for i in counts:
#         error = abs(i-ground_truth)/ground_truth
#         error_list.append(error)
#     mean_error = statistics.mean(error_list)   

#     # Summary
#     print("-----Summary------")
#     print("Ground Truth: ",ground_truth)
#     print("Prediction: ", format(ave_pred, ' .3f'))
#     print("Threshold: ", format(threshold, '.3f'))
#     print("Mean error: ", format(mean_error, ".3f"))
#     print("Max count: ",max(frame_count))
#     print("Min count: ",min(frame_count))
#     print("# Frames: ", len(Frame_number))
#     print("\n")

def smooth(x,y,resolution):
    model=make_interp_spline(x, y)
    x_s=np.linspace(min(x),max(x),resolution)
    y_s=model(x_s)
    return x_s, y_s      

def plot_loss(epoch,loss,val_epochs,metric):
    res = 500

    fig, ax1 = plt.subplots()

    # LOSS
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Loss', color=color)

    # Smooth data
    epochs_s, loss_s = smooth(epoch,loss,res)
    ax1.plot(epochs_s, loss_s, color=color)
    
    ax1.tick_params(axis='y', labelcolor=color)

    # Metric
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('mAP', color=color)  # we already handled the x-label with ax1
    val_epochs_s, metric_s = smooth(val_epochs,metric,res)


    z = np.polyfit(val_epochs,metric,5)
    p = np.poly1d(z)


    ax2.plot(val_epochs, p(val_epochs), color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

    # # Show legend
    # plt.legend()
    # # Add grid 
    # plt.grid()
    # # Show plot
    # plt.show()

def takeFirst(elem):
    return elem[0]

def takeThird(elem):
    return elem[2]

def takeSecond(elem):
    return elem[1]


def takeFifth(elem):
    return elem[4]


def create_2d_array(lst):
    # Determine the number of rows and columns in the 2D array
    num_rows = len(lst) // 5
    num_cols = 5

    # Reshape the 1D list into a 2D array with the specified number of rows and columns
    arr_2d = np.array(lst[:num_rows*num_cols]).reshape(num_rows, num_cols)

    return arr_2d

def normalize_DRB(Z,top,bottom):
    '''
    Args:
        Z::[array_like]

    Return:
        Z_norm::[array_like]
            An array of the normalised values of Z within the bound 2.14 and 5.78
    '''
    max_z = np.max(Z)
    min_z = np.min(Z)
    # max_z = 1.1997789183067033e-06
    # min_z = 2.080361024513692e-09
    Z = np.array(Z)
    # max_z =  1.8656478117607638e-06
    # min_z =  2.826342988779994e-09
    tmax = top
    tmin = bottom
    z_norm = (tmax-tmin)*(Z - min_z)/(max_z-min_z) + tmin

    # z_norm = np.log2(Z)
    return z_norm

    # max_z = np.max(Z)
    # min_z = np.min(Z)
    # A = (top-bottom)
    # B = Z - min_z
    # C = np.multiply(A,B)
    # D = 1/(max_z-min_z)
    # z_norm = np.multiply(C,D)

# Inputs
def LDTS(txt_path,threshold,top,bottom ,return_data=False): 
    ''' Args:

            txt_path::[str]
                Path to the txt with corresponding to data from frame from video

            threshold::[float]
                Counting threshold value - this value is scaled according to the density

            
            top::[float]
                The upper limit to the normalization - normalize_DRB()
                
            bottom::[float]
                The lower limit to the normalization - normalize_DRB()

            return_data::[Bool]
                If return_data == True, returns all data
                else only return count
        
        Returns:

            count::[int]
                The count estimate for the frame (image)
            
            (if return_data==True)    
            X_center,X_center,P,Ts,Z_norm::[array]
                
                X_center,Y_center center of bb coordinates
                P - classification probability values
                Ts - Scaled counting thresholds
                Z_norm - normalized density values 
    '''
    # variable to only run the first loop once
    first = True
    # Process detection data
    results=get_results(txt_path)


    num_rows = len(results) // 5
    num_cols = 5

    # Reshape the 1D list into a 2D array with the specified number of rows and columns
    detection_data = np.array(results[:num_rows*num_cols]).reshape(num_rows, num_cols)

    # Filter out P < 0.05
    detection_data = detection_data[detection_data[:,4 ] > 0.2]
    # Get center points of bounding box
   
    X1 = detection_data[:,0]
    Y1 = detection_data[:,1]
    X2 = detection_data[:,2]
    Y2 = detection_data[:,3]
    P  = detection_data[:,4]
    
    # Calculate the area of each detection
    # box_area = calculate_box_area(X1,Y1,X2,Y2)

    # mask = box_area > 3928/1000

    # X1 = X1[mask]
    # Y1 = Y1[mask]
    # X2 = X2[mask]
    # Y2 = Y2[mask]
    # P  = P[mask]

   




  
    # Get center of each bb
    X_center = np.add(X1,X2)/2
    Y_center = np.add(Y1,Y2)/2

    # Get density distribution f() and evaluate at the model f(X,Y)
    model = KDEMultivariate([X_center,Y_center],'cc',bw ="normal_reference")

    bw = model.bw
   
 
    Z = model.pdf([X_center ,Y_center])
    Z_norm  = normalize_DRB(Z,top,bottom)
    
    # Scale counting threshold
      #array of scaled thresholds
    #T = np.full_like(Z_norm,(1/threshold))

    # for i in range(len(Z_norm)):

    #     density = Z_norm[i]

    #     if density < 3:
    #          Z_norm[i] = 1

    Z_r = np.reciprocal(Z_norm)

    


    Ts = np.multiply(Z_r,threshold)  
   

    # for z in Z_norm:Z
        

    # Get the count
    count = np.count_nonzero(P >= Ts)

    if return_data:
        return X_center,Y_center,P,Ts,Z_norm
    else:
        return count


def base_line(txt_path,Ts):
    ''' Args:

            txt_path::[str]
                Path to the txt with corresponding to data from frame from video
            threshold::[float]
                Counting threshold value - this value is scaled according to the density        
        Returns:

            count::[int]
                The count estimate for the frame (image)           
    '''

    # Process detection data
    results=get_results(txt_path)

    num_rows = len(results) // 5
    num_cols = 5
    # Reshape the 1D list into a 2D array with the specified number of rows and columns
    detection_data = np.array(results[:num_rows*num_cols]).reshape(num_rows, num_cols)
    # Filter out P < 0.05
    detection_data = detection_data[detection_data[:,4 ] > 0.2]
    # Get center points of bounding box
    P  = detection_data[:,4]
    # Get the count
    count = np.count_nonzero(P > Ts)
    return count

def video_counts_func(video_path,max_frames,upper,lower,ts):
    '''
    Gets count of all frames in a video and calculated the average count

    Args:
        video_path::[str]
            Path to video frames
        max_frames::[int]
            The number of frames to use when calculating the average 
            For development this is set to +-15 frames while during testing this should be set to >200
        upper::[float]
            Upper normalisation bound
        lower::[float]
            lower normalisation bound
    Returns:
        average_count::[int]
            The average sheep count in the video
        bl_average_count::[int]
            The baseline average sheep count in the video
    
    '''
    limit = 0
    video_counts = []
    bl_video_counts = []
    norm_upper = upper
    norm_lower = lower
    threshold = ts
    for frame in os.listdir(video_path):

        if limit < max_frames:
            frame_path = os.path.join(video_path,frame)
            count = LDTS(frame_path,threshold,norm_upper, norm_lower)
            bl_count = base_line(frame_path,threshold)
            bl_video_counts.append(bl_count)
            video_counts.append(count)
            limit +=1
        else:
            break
        
    average_count = stats.mean(video_counts)
    bl_average_count = stats.mean(bl_video_counts)

    return average_count,bl_average_count

def ground_truth(vid):
    '''
    Get ground truth from video name
    
    Args:
        vid::[str]
            Name of video 
    Return
        gt::[int]
            The ground truth count of a video
            
    '''
    gt = int(vid.split("-")[1])
        # Try to remove space before gt (( eg: DJI_0808 -" "1200))
    try:
        gt = int(gt.split(" ")[1])

    except: 
        # Does nothing
        hold = 1
    return gt



def objective_func(params,num_frames,ts):
    '''
    Function to be minimised - used to optimise params
    Args:
        params::[array]
            Array containing the params x0 and x1
        num_frames::[int]
            Number of frames to use 
    Returns
        mae::[float]
            the mean average error (value to be minimised)
    
    '''

    ignore_files = ["Altitude_videos","Circle_vid","dicts","val",'DJI_0830-522']
    videos_path = r"C:\Users\USER\OneDrive - Stellenbosch University\Masters\Thesis\virtual_envs\coldb_files\four_models\Cascade R-CNN_V3"

    frame_limit = num_frames
    video_counts = []
    video_groundtruths =[]
    bl_video_counts = []
    video_errors = []
    bl_video_errors = []
    use_counts = []
    algorithm_errors_list = []
    gts = []
    for video in os.listdir(videos_path):
        if video not in ignore_files:
    
            gt = ground_truth(video)
            video_groundtruths.append(gt)
            video_path = os.path.join(videos_path,video)
            average_count,bl_average_count  = video_counts_func(video_path,frame_limit,params[0],params[1],ts)
            
            # if (average_count and bl_average_count) < 400:
            #     count = bl_average_count
            # else:
            count = average_count

            
            video_counts.append(average_count)
            use_counts.append(count)
            bl_video_counts.append(bl_average_count)

            error = abs(count-gt)
            bl_error =  abs(bl_average_count-gt)
            algorithm_errors = abs(count-gt)
            bl_video_errors.append(bl_error)
            video_errors.append(error)
            algorithm_errors_list.append(algorithm_errors)
            gts.append(gt)

    mae = MAE(video_counts,video_groundtruths)
    bl_mae = MAE(bl_video_counts,video_groundtruths)


    return algorithm_errors_list, mae, video_errors,bl_mae, bl_video_errors,gts
    


# def objective_function(params):
#     '''
#     Function to be minimised - used to optimise params
#     Args:
#         params::[array]
#             Array containing the params x0 and x1
    
#     Returns
#         mae::[float]
#             the mean average error (value to be minimised)
    
#     '''

#     ignore_files = ["Altitude_videos","Circle_vid","dicts","val",'DJI_0830-522']
#     videos_path = r"C:\Users\USER\OneDrive - Stellenbosch University\Masters\Thesis\virtual_envs\coldb_files\four_models\Cascade R-CNN_V3"

#     frame_limit = 2
#     video_counts = []
#     video_groundtruths =[]
#     bl_video_counts = []
#     video_errors = []
#     bl_video_errors = []
#     use_counts = []
#     algorithm_errors_list = []
#     gts = []
#     for video in os.listdir(videos_path):
#         if video not in ignore_files:
#             gt = ground_truth(video)
#             video_groundtruths.append(gt)
#             video_path = os.path.join(videos_path,video)
#             average_count,bl_average_count  = video_counts_func(video_path,frame_limit,params[0],params[1])
            
#             if (average_count and bl_average_count) < 350:
#                 count = bl_average_count
#             else:
#                 count = average_count

            
#             video_counts.append(average_count)
#             use_counts.append(count)
#             bl_video_counts.append(bl_average_count)

#             error = abs(count-gt)
#             bl_error =  abs(bl_average_count-gt)
#             algorithm_errors = abs(count-gt)
#             bl_video_errors.append(bl_error)
#             video_errors.append(error)
#             algorithm_errors_list.append(algorithm_errors)
#             gts.append(gt)

#     mae = MAE(video_counts,video_groundtruths)

#     return mae
   

def LDTS_with_no_txt(detection_data,threshold,top,bottom ,return_data=False): 
# Filter out P < 0.05
    detection_data = detection_data[detection_data[:,4 ] > 0.2]
    # Get center points of bounding box
   
    X1 = detection_data[:,0]
    Y1 = detection_data[:,1]
    X2 = detection_data[:,2]
    Y2 = detection_data[:,3]
    P  = detection_data[:,4]
    
    # Get center of each bb
    X_center = np.add(X1,X2)/2
    Y_center = np.add(Y1,Y2)/2

    # Get density distribution f() and evaluate at the model f(X,Y)
    model = KDEMultivariate([X_center,Y_center],'cc',bw ="normal_reference")

    bw = model.bw
   
  
    Z = model.pdf([X_center ,Y_center])
    Z_norm  = normalize_DRB(Z,top,bottom)
    
    # Scale counting threshold
      #array of scaled thresholds
    #T = np.full_like(Z_norm,(1/threshold))
    Z_r = np.reciprocal(Z_norm)
    Ts = np.multiply(Z_r,threshold)  
   

    # for z in Z_norm:Z
        

    # Get the count
    count = np.count_nonzero(P > Ts)

    if return_data:
        return X_center,Y_center,P,Ts,Z_norm
    else:
        return Ts
    



def calculate_box_area(x1, y1, x2, y2):
    """
    Calculates the area of a box given its coordinates (x1, y1, x2, y2).
    """
    length = x2 - x1
    width = y2 - y1
    area = length * width
    return area