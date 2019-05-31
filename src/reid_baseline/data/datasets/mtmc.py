import os
import random
import pandas as pd
from sklearn import preprocessing

def process_trn():
    df = pd.DataFrame()
    trn_imgs, vids, cids = [], [], []
    trn_dir = '/nfs/project/dataset/AI_city_2019/reid/train'
    for vid in os.listdir(trn_dir):
        v_path = os.path.join(trn_dir, vid)
        for img_name in os.listdir(v_path):
            img_path = os.path.join(v_path, img_name)
            #print(img_path)
            ls = img_name.split('_')
            vid, cid = ls[1], ls[0][1:]
            trn_imgs.append(img_path)
            vids.append(vid)
            cids.append(cid)
    df['img_path'] = trn_imgs
    df['vid'] = vids
    df['cid'] = cids

    vid_le = preprocessing.LabelEncoder()
    cid_le = preprocessing.LabelEncoder()

    df['v'] = vid_le.fit_transform(df['vid'].tolist())
    df['c'] = cid_le.fit_transform(df['cid'].tolist())

    print(len(set(df['v'])))
    return df



df = process_trn()
trn_df = df[df['v']<128]
val_df = df[df['v']>=128]


vid_le = preprocessing.LabelEncoder()
cid_le = preprocessing.LabelEncoder()
val_df['v'] = vid_le.fit_transform(val_df['v'].tolist())
val_df['c'] = cid_le.fit_transform(val_df['c'].tolist())


trn_set = trn_df[['img_path', 'v', 'c']].values.tolist()
val_set = val_df[['img_path', 'v', 'c']].values.tolist()

random.shuffle(val_set)
q,g =val_set[:160],val_set[160:]

if __name__ == '__main__':
    print(df.head())
    print(len(df))
    print(len(trn_set),len(val_set))
    print(trn_set[:7])
    print(val_set[:7])
