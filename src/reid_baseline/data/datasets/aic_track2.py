# encoding: utf-8
import glob
import re
import os
import os.path as osp
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn import preprocessing
from .bases import BaseImageDataset
from .mtmc import trn_set, q,g

class aic_track2(BaseImageDataset):
    dataset_dir = 'data/'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(aic_track2, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_test')
        self.cv_df = pd.read_csv(os.path.join(root,'track2_cv.csv'))
        #self._check_before_run()

        #train = self._process_train(self.cv_df, 1)
        #query, gallery = self._process_val(self.cv_df, 1, 500)
        train = trn_set
        query, gallery = q,g
        query_test, gallery_test = self._process_test()
        if verbose:
            print("=> AIC Track2 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.query_test = query_test
        self.gallery_test = gallery_test

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        #self.num_query_test_pids, self.num_query_test_imgs, self.num_query_test_cams = self.get_imagedata_info(self.query_test)
        #self.num_gallery_test_pids, self.num_gallery_test_imgs, self.num_gallery_test_cams = self.get_imagedata_info(self.gallery_test)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_val(self, df, fold, num_query):
        val_df = df[df['fold'] == fold]
        val_df = shuffle(val_df.copy())
        val_df['imgPath'] = val_df['imgName'].apply(lambda x: os.path.join(self.train_dir, x))

        vid_le = preprocessing.LabelEncoder()
        cid_le = preprocessing.LabelEncoder()

        val_df['vid'] = vid_le.fit_transform(val_df['vehicleID'].tolist())
        val_df['cid'] = cid_le.fit_transform(val_df['cameraID'].tolist())

        val_df = val_df[['imgPath', 'vid', 'cid']]
        dataset = val_df.values.tolist()

        num_query = float(len(dataset))*1052/(18290+1052)
        num_query = int(num_query)
        query, gallery = dataset[:num_query], dataset[num_query:]

        return query, gallery

    def _process_trn(self):
        trn_imgs, vids, cids = [], [], []
        trn_dir = '/nfs/project/dataset/AI_city_2019/reid/train'
        for vid in os.listdir(trn_dir):
            v_path = os.path.join(trn_dir, vid)
            for img_name in os.listdir(v_path):
                img_path = os.path.join(v_path, img_name)
                ls = img_path.split('_')
                vid, cid = int(ls[1]), int[ls[0][1:]]
                trn_imgs.append(img_path)
                vids.append(vid)
                cids.append(cid)
        random.shuffle(trn_imgs)
        


    def _process_train(self, df, fold):
        train_df = df[df['fold'] != fold]
        train_df['imgPath'] = train_df['imgName'].apply(lambda x: os.path.join(self.train_dir, x))
        
        vid_le = preprocessing.LabelEncoder()
        cid_le = preprocessing.LabelEncoder()
        
        train_df['vid'] = vid_le.fit_transform(train_df['vehicleID'].tolist())
        train_df['cid'] = cid_le.fit_transform(train_df['cameraID'].tolist())
        
        train_df = train_df[['imgPath', 'vid', 'cid']]
        dataset = train_df.values.tolist() 

        return dataset

    #def _process_test(self):
    #    query_imgs = [os.path.join(self.query_dir, img) for img in os.listdir(self.query_dir)]
    #    gallery_imgs = [os.path.join(self.gallery_dir, img) for img in os.listdir(self.gallery_dir)]
    #    query = [[img, -1, -1] for img in query_imgs]
    #    gallery = [[img, -1, -1] for img in gallery_imgs]

    #    return query, gallery

    def _process_test(self):
        gallery_imgs = []
        test_dir = '/nfs/project/dataset/AI_city_2019/reid/eval_cropped_imgs/c027'
        for f in os.listdir(test_dir):
            p = os.path.join(test_dir, f)
            gallery_imgs.append(p)
        query_imgs = [gallery_imgs[0]]
        query = [[img, -1, -1] for img in query_imgs]
        gallery = [[img, -1, -1] for img in gallery_imgs]
        return query, gallery 

    #def _process_test(self):
    #    gallery_imgs = []
    #    test_dir = '/nfs/project/dataset/AI_city_2019/reid/eval_cropped_imgs'
    #    for f in os.listdir(test_dir):
    #        p = os.path.join(test_dir, f)
    #        gallery_imgs.append(p)
    #    query_imgs = [gallery_imgs[0]]
    #    query = [[img, -1, -1] for img in query_imgs]
    #    gallery = [[img, -1, -1] for img in gallery_imgs]
    #    return query, gallery

if __name__ == '__main__':
    df = process_trn() 
    print(df.head())
