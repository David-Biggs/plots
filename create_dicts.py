import numpy as np
import os
import json

from functions import *
global thresholds
models =r"C:\\Users\\USER\\OneDrive - Stellenbosch University\\Masters\\Thesis\\virtual_envs\\coldb_files\\four_models\\"

print(models)
counter = 0
def cr_thr():
    global thresholds
    thresholds = np.linspace(0.02,0.98,25)
    return thresholds

thresholds = cr_thr()
#thresholds = [0.2,0.5,0.8]                          
model_dict ={}

for model in os.listdir(models):
    print("Enter model loop")
    print(model)
    if model == "Cascade R-CNN" or model =="Cascade R-CNN_V2" or model =="Cascade R-CNN_V3" or model =='cascade_rcnn':
        print(model)
        vid_dict = {} 
        main_dict = {}
        # Loop over all thresholds
        all_thresholds_acc =[]
        all_vid_max_detect =[]
        m_all_vids_list =[]
        total_imgs = 0
    
        videos = os.path.join(models,model)
        for threshold in thresholds:   
            data_dict={}
            video_dict={}
            all_vids_acc = []
            m_vid_det_max = [] 
            # loop through all videos
            for video in os.listdir(videos):
                m_vid_acc = 0
                vid_accs = []
                
                vid_det_max =[]
                if video != '.ipynb_checkpoints' and video != "DJI_0715 - 60" and video != "DJI_0762 - 577" and video!='dicts':    
                    # Loop throuhg all .txt result files in frames dir
                
                    # if str(video) == "DJI_0713 - 60" or str(video) == "DJI_0715 - 60" or str(video) == "DJI_0734 - 1" or str(video) == "DJI_0731 - 47":
                    #     print("Skipped: ", video)
                    # else:
                    vid_path = os.path.join(videos,video)
                    #frames = os.path.join(vid_path,'Frames')
                    for txt in os.listdir(vid_path):
                        if txt.endswith(".txt"):
                            # Get global path to current .txt file
                            txt_path = os.path.join(vid_path,txt)
                        
                            results = get_results(txt_path)
                            probs = get_probs(results)
                            PC = get_count(probs, threshold)
                            
                            GT = get_GT(txt)
                            
                            
                            img_acc = get_acc(GT, PC, txt,True)
                            vid_det_max.append(len)
                            vid_accs.append(img_acc)
                            total_imgs += 1
                            
                    #Mean acc for current video
                    if len(vid_accs)>1:
                        m_vid_acc = statistics.mean(vid_accs)
                        all_vids_acc.append(m_vid_acc)
                    else:
                        print("Only one element")
                    # Score video acc in dict for plotting later
                    if not video in vid_dict:
                        vid_dict[video] = [m_vid_acc]
                    else:
                        vid_dict[video].append(m_vid_acc)
                    
                    name = video.split("-")
                    data_dict["count"] = name[1]
                    data_dict["error"] = m_vid_acc
                    video_dict[video] = [int(name[1]), float(m_vid_acc)]
                    #print(video_dict)
                    counter += 1

        
            main_dict[threshold] = video_dict
            m_all_vids = statistics.mean(all_vids_acc)
            m_all_vids_list.append(m_all_vids)
            all_thresholds_acc.append(m_all_vids)
            model_dict[model] = all_thresholds_acc
            print("Threshold: ", threshold)
            print("Error: ",m_all_vids)


            dicts_root = r"C:\\Users\\USER\\OneDrive - Stellenbosch University\\Masters\\Thesis\\virtual_envs\\coldb_files\\four_models\\" + model + "\\dicts"
            #Save dicts
            #Main dict
            main_dict_file_name = 'main_dict.json'
            main_dict_path = os.path.join(dicts_root,main_dict_file_name)
            with open(main_dict_path, "w") as outfile:
                json.dump(main_dict, outfile)

            #model_dict
            model_dict_file_name = 'model_dict.json'
            model_dict_path = os.path.join(dicts_root,model_dict_file_name)
            with open(model_dict_path, "w") as outfile:
                json.dump(model_dict, outfile)     
print(total_imgs)



