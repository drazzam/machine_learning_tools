from __future__ import print_function
import pandas as pd
import shutil
import os
import sys

labels = pd.read_csv(r'C:\Users\Ahmed Azzam\Downloads\Compressed\Training_Set\Training_Set\class.csv')

train_dir =r'C:\Users\Ahmed Azzam\Downloads\Compressed\Training_Set\Training_Set\Training'
DR = r'C:\Users\Ahmed Azzam\Downloads\Compressed\Training_Set\Training_Set\DR'
if not os.path.exists(DR):
    os.mkdir(DR)

for filename, class_name in labels.values:
    # Create subdirectory with `class_name`
    if not os.path.exists(DR + str(class_name)):
        os.mkdir(DR + str(class_name))
    src_path = train_dir + '/'+ str(filename) + '.png'
    dst_path = DR + str(class_name) + '/' + str(filename) + '.png'
    try:
        shutil.copy(src_path, dst_path)
        print("sucessful")
    except IOError as e:
        print('Unable to copy file {} to {}'
              .format(src_path, dst_path))
    except:
        print('When try copy file {} to {}, unexpected error: {}'
              .format(src_path, dst_path, sys.exc_info()))
