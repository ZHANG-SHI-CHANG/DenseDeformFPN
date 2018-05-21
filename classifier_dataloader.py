import numpy as np
import cv2

import os
import glob

class DataLoader:
    def __init__(self,root=None,batch=10,shuffle=True):
        self.root = root
        
        self.num_data = 0
        self.classes = 5
        
        self.datas = None
        self.labels = None
        
        self.batch = batch
        self.shuffle = shuffle
        
        self.input_height = 224
        self.input_width = 224
    
    def to_list(self):
        datas = []
        labels = []
        for i,file_name in enumerate(glob.glob( os.path.join(self.root,'*') )):
            label = int(file_name.split('\\')[-1])
            for j,image_name in enumerate(glob.glob( os.path.join(file_name,'*.png') )):
                try:
                    datas.append(image_name)
                    labels.append(label)
                except:
                    pass
        
        return datas,labels
    
    def prepare(self):
        self.datas,self.labels = self.to_list()
        
        assert len(self.datas)==len(self.labels)
        self.num_data = len(self.datas)
        
        self._count_in_epoch = 0
        self._count_epoch = 0
    
    def __len__(self):
        return len(self.datas)//self.batch
        
    def to_one_numpy(self,data,label):
        try:
            data = cv2.imread(data).astype(np.float)
        except:
            print('error image {}'.format(data))
        data = cv2.resize(data,(self.input_height,self.input_width))
        try:
            c = data.shape[2]
        except:
            data = data[:,:,np.newaxis]
            data = np.concatenate((data,data,data),axis=2)
        return data[np.newaxis,:,:,:].astype(np.float),np.array([label]).astype(np.float)
    
    def to_one_hot(self,labels):
        _labels = np.zeros((labels.shape[0],self.classes))
        _labels[np.arange(labels.shape[0]),labels] = 1
        return _labels
    
    def to_numpy(self,datas,labels):
        assert len(datas)==len(labels)
        if len(datas)==1:
            return self.to_one_numpy(datas[0],labels[0])
        elif len(datas)>1:
            data,label = self.to_one_numpy(datas[0],labels[0])
            for i in range(1,len(datas)):
                _data,_label = self.to_one_numpy(datas[i],labels[i])
                data,label = np.concatenate((data,_data),axis=0),np.concatenate((label,_label),axis=0)
            return data,label
    
    def next(self):
        self.prepare()
        while True:
            if self._count_epoch==0 and self._count_in_epoch==0 and self.shuffle:
                self._datas,self._labels = [],[]
                permutation = np.arange(self.num_data)
                np.random.shuffle(permutation)
                for i in permutation:
                    self._datas.append(self.datas[i])
                    self._labels.append(self.labels[i])
            if self._count_in_epoch+self.batch>self.num_data:
                self._count_epoch += 1
                
                start = self._count_in_epoch
                self._count_in_epoch = 0
                
                _rest = self.num_data - start
                rest_datas,rest_labels = self._datas[start:self.num_data],self._labels[start:self.num_data]
                
                if self.shuffle:
                    self._datas,self._labels = [],[]
                    permutation = np.arange(self.num_data)
                    np.random.shuffle(permutation)
                    for i in permutation:
                        self._datas.append(self.datas[i])
                        self._labels.append(self.labels[i])
                
                start = self._count_in_epoch
                self._count_in_epoch = self.batch-_rest
                end = self._count_in_epoch
                
                new_datas,new_labels = self._datas[start:end],self._labels[start:end]
                datas,labels = rest_datas+new_datas,rest_labels+new_labels
            else:
                start = self._count_in_epoch
                self._count_in_epoch += self.batch
                end = self._count_in_epoch
                datas,labels = self._datas[start:end],self._labels[start:end]
            
            datas,labels = self.to_numpy(datas,labels)
            yield datas,labels

if __name__=='__main__':
    dataloader = DataLoader()
    for datas,labels in dataloader.next():
        print(datas.shape,labels.shape)