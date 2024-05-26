import os,csv
import shutil

dirpath = "Training_Dataset/training_voice_data"

# for i in os.listdir(dirpath):    
    # print(i[:-4])

with open('Training_Dataset/training_datalist.csv') as f:
    datalist_dict = csv.DictReader(f)
    list_of_dict = list(datalist_dict)
    #print(list_of_dict)


for i in range(len(list_of_dict)):
    #old_loc = f'Training_Dataset/training_voice_data/{i["ID"]}.wav'
    if(list_of_dict[i]['Disease category']=='1'):
        list_of_dict[i]['Disease category']='1'
    if(list_of_dict[i]['Disease category']=='2'):
        list_of_dict[i]['Disease category']='2'
    if(list_of_dict[i]['Disease category']=='3'):
        list_of_dict[i]['Disease category']='2'
    if(list_of_dict[i]['Disease category']=='4'):
        list_of_dict[i]['Disease category']='1'
    if(list_of_dict[i]['Disease category']=='5'):
        list_of_dict[i]['Disease category']='0'

with open('Training_Dataset/training_datalist_resign.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list_of_dict[0].keys())
    writer.writeheader()
    for i in list_of_dict:
        writer.writerow(i)