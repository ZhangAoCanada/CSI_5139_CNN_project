import pickle

prefix = "train_2015_mrcnn/"

name_len = 6

for i in range(10):
    img_count = str(i)
    zero_len = name_len - len(img_count)
    img_name = (zero_len * "0") + img_count + "_10"

    with open(prefix + img_name + ".pickle", "rb") as f:
        b = pickle.load(f)
    
    print(b["scores"].shape)