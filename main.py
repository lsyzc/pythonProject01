from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, train_val, image_dir, labels_dir):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.train_val = train_val
        self.image_path = os.path.join(self.root_dir, self.train_val, self.image_dir)
        self.image_label = os.path.join(self.root_dir, self.train_val, self.labels_dir)
        self.image_name = os.listdir(self.image_path)
        self.label_name = os.listdir(self.image_label)
        self.images_list = [os.path.join(self.image_path, i) for i in self.image_name]
        self.label_list = [os.path.join(self.image_label, i) for i in self.label_name]

    def __getitem__(self, idx):
        image = self.images_list[idx]
        print(image)
        image = Image.open(image)
        image_label_path = self.label_list[idx]
        with open(image_label_path) as f:
            label = f.readline()
            print(type(label))

        return {"image": image, "label": label}

    def __len__(self):
        return len(self.image_name)


my_data = MyData("Data_exercise", "train", "ants_image", "ants_label")
seccond_data = MyData("Data_exercise","train","bees_image","bees_label")
data = my_data+seccond_data
data.__getitem__(124)["image"].show()
print("mydata's length:"+str(my_data.__len__())+"\nsecond 's length:"+str(seccond_data.__len__())+"\ndata 's length"+str(data.__len__()))

