import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
from warnings import simplefilter
import pandas as pd

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
"""
Files:
    data_key.json: 
        [id_raw, split, text, img_names, num_imgs, category_name, label]
    pp_images_{grid_model}_{split}.npz
        Grid-level image features for each model (ResNet, Xception, EfficientNet)
        and for each split (train, dev, test)
    poi_{split}_ids.txt
        IDs of tweets with image content for each split (train, dev, test)   

Example:
  data_loader = DataLoader("AvgAll","../data/", "RESNET")  
  splits, class_weight_dict = data_loader.load_data_splits()
"""

grid_models = {
    "RESNET": "res",
    "XCEPTION": "xc",
    "EFFICIENTNET": "effn"
}


class DataLoader():
    def __init__(self, dataset, data_path, model, num_labels=8, testing=0):
        """
        dataset (str): AvgAll (all dataset using average image for tweets without image),
                       SampleWImg (only tweets with image)
        data_path (str): data path of the folder containing the data files
        model (str): model names as in multimodal_dict (mm_main.py)
        num_labels (int): number of labels
        testing (int): set to 1 for testing mode
        """
        self.split_ids = {}
        self.data = []
        # Load data
        self.DATASET = dataset
        self.DATA = data_path
        self.data = pd.read_json(self.DATA + "data_key.json", orient="records")
        self.MODEL = model
        self.NUM_LABELS = num_labels
        self.TESTING = testing
        self.img_index = {s: {} for s in ["train", "dev", "test"]}
        self.IMGS = self.DATA + "pp_images_{}_{}.npz"
        feats_tr = np.load(self.IMGS.format(grid_models[self.MODEL], "train"))["arr_0"]
        self.feats_tr_mean = np.mean(feats_tr, axis=0)

    def load_data_splits(self, split_names=None):
        """
        Load text and image content
        split_names (list <str>): split names to load (train,dev,test)
        Returns:
            splits (dict): splits[split][imgs/text/lbls]
            class_weight (dict): class weight for training
        """
        class_weight_dict = {}
        # SPLITS
        if split_names is None:
            split_names = ["train", "dev", "test"]
        splits = {x: {} for x in split_names}
        for split in splits:
            IDS = self.DATA + "poi_{}_ids.txt".format(split)  # split: train, dev, test
            # load img_ids
            split_ids = []
            for line in open(IDS):
                split_ids.append(int(line.strip()))
            self.img_index[split] = {id_: idx for idx, id_ in enumerate(split_ids)}
            if self.TESTING == 1:
                if split == "test":
                    split_ids = [792506233748021248, 901583547898540033, 823946411196563456, 925860602496634881,
                                 832324417334054912, 935778076562034688, 851832319371509760, 961343897375997954]
                else:
                    split_ids = random.sample(split_ids, 10)

            # GRID-LEVEL
            if self.TESTING == 1:
                idxs = [self.img_index[split][i] for i in split_ids]
                splits[split]["imgs"] = np.load(self.IMGS.format(grid_models[self.MODEL], split))["arr_0"][idxs, :]
            else:
                splits[split]["imgs"] = np.load(self.IMGS.format(grid_models[self.MODEL], split))["arr_0"]

            if self.DATASET == "AvgAll":
                # ASSIGN AVG FEATURES TO TWEETS WITHOUT IMAGES
                split_ids_nimg = set(self.data[self.data.split == split.upper()].id_raw.values) - set(split_ids)
                if self.TESTING == 1:
                    split_ids_nimg = random.sample(split_ids_nimg, 100)

                splits[split]["imgs"] = np.concatenate([splits[split]["imgs"],
                                                        np.asarray([self.feats_tr_mean] * len(split_ids_nimg))])
                split_ids += split_ids_nimg

            print(self.MODEL, split, splits[split]["imgs"].shape, "split_ids", len(split_ids))

            # labels in the same order
            df = self.data.set_index("id_raw").loc[split_ids]
            print("Len", split, ":", len(df))
            # vectorise labels
            self.NUM_LABELS = 8
            labels_enc = np.zeros((len(df), self.NUM_LABELS))
            for i, y in enumerate(df.label.values):
                labels_enc[i][y] = 1
            splits[split]["lbls"] = labels_enc
            # get text
            splits[split]["text"] = df.text.values
            if split == "train":
                # class_weight
                classes = np.unique(df.label.values)
                class_weight = compute_class_weight("balanced", classes, df.label.values)
                class_weight_dict = dict(enumerate(class_weight))
            self.split_ids[split] = split_ids
        return splits, class_weight_dict

    def load_text_split(self, dataset):
        """
        Load text content only
        dataset (str): tweets with/without image - SampleWImg, SampleWOImgs
        Returns:
            splits (dict): splits[split][text/lbls]
            class_weight (dict): class weight for training
        """
        class_weight_dict = {}
        splits = {x: {} for x in ["train", "dev", "test"]}
        for split in splits:
            if dataset == "SampleWImg":
                # SPLITS
                IDS = self.DATA + "poi_{}_ids.txt".format(split)  # split: train, dev, test
                # load img_ids
                split_ids = []
                for line in open(IDS):
                    split_ids.append(int(line.strip()))
                if self.TESTING == 1:
                    split_ids = random.sample(split_ids, 100)
                split_data = self.data.set_index("id_raw").loc[split_ids]
            elif dataset == "SampleWOImgs":
                # SPLITS
                IDS = self.DATA + "poi_{}_ids.txt".format(split)  # split: train, dev, test
                # load img_ids
                split_ids = []
                for line in open(IDS):
                    split_ids.append(int(line.strip()))
                split_ids_nimg = set(self.data[self.data.split == split.upper()].id_raw.values) - set(split_ids)
                if self.TESTING == 1:
                    split_ids_nimg = random.sample(split_ids_nimg, 100)
                split_data = self.data.set_index("id_raw").loc[split_ids_nimg]
            else:
                split_data = self.data[self.data["split"] == split.upper()]
            print("Len", split, ":", len(split_data))
            text = split_data.text.values
            cat = split_data.label.values
            labels_enc = np.zeros((len(cat), 8))
            for i, y in enumerate(cat):
                labels_enc[i][y] = 1
            if split == "train":
                # class wight
                classes = list(set(cat))
                class_weight = compute_class_weight("balanced", classes, cat)
                class_weight_dict = dict(enumerate(class_weight))
            splits[split]["text"] = text
            splits[split]["lbls"] = labels_enc
        return splits, class_weight_dict
