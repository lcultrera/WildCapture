import os
import random
import shutil
      
class DatasetSplitter:
    def __init__(self, data_dir, train_dir, val_dir, train_perc=0.8):
        self.data_dir = data_dir
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.train_perc = train_perc

    def split_dataset(self):
        for img_class in os.listdir(self.data_dir):
            files = os.listdir(os.path.join(self.data_dir, img_class))
            random.shuffle(files)
            num_files = len(files)
            num_train = int(self.train_perc * num_files)

            train_files = files[:num_train]
            val_files = files[num_train:]

            # Create train and val directories if they don't exist
            os.makedirs(os.path.join(self.train_dir, img_class), exist_ok=True)
            os.makedirs(os.path.join(self.val_dir, img_class), exist_ok=True)

            # Move files to train directory
            for f in train_files:
                src_path = os.path.join(self.data_dir, img_class, f)
                dest_path = os.path.join(self.train_dir, img_class, f)
                shutil.move(src_path, dest_path)

            # Move files to val directory
            for f in val_files:
                src_path = os.path.join(self.data_dir, img_class, f)
                dest_path = os.path.join(self.val_dir, img_class, f)
                shutil.move(src_path, dest_path)

