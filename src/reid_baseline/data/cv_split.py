import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import KFold


with open('../../data/train_label.xml','r') as f:
    tree = ET.fromstring(f.read())


map_df = pd.DataFrame()
imgName,cameraID,vehicleID = [],[],[]

for ele in tree:
    for e in ele:
        imgName.append(e.attrib['imageName'])
        cameraID.append(e.attrib['cameraID'])
        vehicleID.append(e.attrib['vehicleID'])
        
# print(imgName,cameraID,vehicleID)
map_df['imgName'] = imgName
map_df['cameraID'] = cameraID
map_df['vehicleID'] = vehicleID
map_df['fold'] = -1 

vids = np.array(list(set(map_df['vehicleID'].tolist())))


print(len(vids))

kf = KFold(n_splits=5, random_state=2019, shuffle=False)

fold_id = 0
for trn_idx,val_idx in kf.split(vids):
    trn_vid,val_vid = vids[trn_idx], vids[val_idx]
    print(len(trn_vid), len(val_vid))

    map_df.loc[map_df['vehicleID'].isin(set(trn_vid))]
    map_df.loc[map_df['vehicleID'].isin(set(val_vid)),['fold']] = fold_id
    fold_id += 1

print(map_df.head())
for i in range(5):
    print(len(map_df[map_df['fold']==i]))

map_df.to_csv('track2_cv.csv',index=False)
