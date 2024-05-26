import os,csv
import shutil

dirpath = "Training_Dataset/training_voice_data"

# for i in os.listdir(dirpath):    
    # print(i[:-4])

with open('Training_Dataset/training_datalist_resign.csv') as f:
    datalist_dict = csv.DictReader(f)
    list_of_dict = list(datalist_dict)
    #print(list_of_dict)

old_loc = ''
new_loc = ''
for i in list_of_dict:
    old_loc = f'Training_Dataset/training_voice_data/{i["ID"]}.wav'
    if(i['Disease category']=='0'):
        new_loc = f'Training_Dataset/training_voice_data/vd_{i["ID"][-2::-2]}{i["ID"][::-2]}.wav'
        shutil.copyfile(old_loc,new_loc)
        print(f'Copyfile from {old_loc} to {new_loc}')

#print(list_of_dict[0]['ID'] + ' ' + list_of_dict[0]['Disease category'])