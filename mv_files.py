from pathlib import Path
import os
import shutil

root_dir = '/scratch/gpfs/zixuz/argoverse-tracking'
destination = '/scratch/gpfs/zixuz/argoverse-tracking/fl'

def mv(train_path):
    if Path(train_path).is_dir():
        for data_path in Path(train_path).iterdir(): 
            source_folder = os.path.join(data_path, 'stereo_front_left')
            destination_folder = os.path.join(destination, 'val/'+data_path.name)
            
            # make folder 
            Path(destination_folder).mkdir(parents=True, exist_ok=True)
            
            if Path(source_folder).is_dir():
                shutil.move(source_folder, destination_folder)

train_path = os.path.join(root_dir, 'val')
mv(train_path)
             
for i in range(1,5):
    train_path = os.path.join(root_dir, 'train'+str(i))
    mv(train_path)
                
                

    