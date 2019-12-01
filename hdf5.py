import numpy as np
import os
import h5py
import cv2
import glob
def test():
    filename="/home/ljm/gdata/DFDC/init/frames/test.hdf5"
    frame_path_pre="/home/hzh/dfdc_1fps_crop/"
    frame_path_suf="method_A/1003254/1003254_A/1255229_1003254_A_001"
    f=h5py.File(filename,'w')
    # print(frame_path_suf.split("/")[0:-1])
    path=""
    for e in frame_path_suf.split("/")[0:-1]:
        path=os.path.join(path,e)
        f.create_group(path)
    path=os.path.join(frame_path_pre,frame_path_suf)
    file_list=os.listdir(path)
    imgs=[]
    for e in file_list:
        file_path=os.path.join(path,e)
        img=cv2.imread(file_path)
        imgs.append(img)
    imgs=np.array(imgs)
    print(imgs.dtype)
    f.create_dataset(frame_path_suf,data=imgs)
    f[frame_path_suf].attrs['len']=len(file_list)
    f.close()
    f=h5py.File(filename,"r")
    print(f[frame_path_suf][:].shape)
    os.remove(filename)
def get_dirs(root):
    files = glob.glob(root + '/*')
    dir=[x.split('/')[-1] for x in files if os.path.isdir(x) ]
    return dir
def full_convert(root):
    dir=['method_A','method_B']
    filename = "/home/ljm/gdata/DFDC/init/frames/fps1_crop256.hdf5"
    if os.path.exists(filename):
        os.remove(filename)
    f = h5py.File(filename, 'w')
    for e in dir:
        f.create_group(e)
        # path=os.path.join(root,e)
        sub_dir=get_dirs(os.path.join(root,e))
        for s in sub_dir:
            f.create_group(os.path.join(e,s))
            # path=os.path.join(path,s)
            sub_sub_dir=get_dirs(os.path.join(root,e,s))
            for t in sub_sub_dir:
                f.create_group(os.path.join(e,s,t))
                # path = os.path.join(path, t)
                datadir=get_dirs(os.path.join(root,e,s,t))
                for d in datadir:
                    filelist=glob.glob(os.path.join(root,e,s,t,d)+ '/*.png')
                    imgs=[cv2.imread(path) for path in filelist]
                    imgs=np.array(imgs)
                    f.create_dataset(os.path.join(e,s,t,d),data=imgs)
                    f[os.path.join(e,s,t,d)].attrs['len']=len(filelist)
    e='original_videos'
    f.create_group(e)
    sub_dir = get_dirs(os.path.join(root, e))
    for s in sub_dir:
        f.create_group(os.path.join(e, s))
        # path=os.path.join(path,s)
        datadir = get_dirs(os.path.join(root, e, s))
        for d in datadir:
            filelist = glob.glob(os.path.join(root, e, s, d) + '/*.png')
            imgs = [cv2.imread(path) for path in filelist]
            imgs = np.array(imgs)
            f.create_dataset(os.path.join(e, s, d), data=imgs)
            f[os.path.join(e, s, d)].attrs['len'] = len(filelist)
    f.close()
def read_test():
    filename = "/home/ljm/gdata/DFDC/init/frames/fps1_crop256.hdf5"
    frame_path_suf = "method_A/1003254/1003254_A/1255229_1003254_A_001"
    f = h5py.File(filename,"r")
    # print(f[frame_path_suf][:])
    print(f[frame_path_suf].attrs['len'])
    # cv2.imwrite("test.png",f[frame_path_suf][0])
    path_to_file='/home/ljm/gdata/DFDC/init/video/fb_dfd_release_0.1_final/three/train.csv'
    with open(path_to_file, "r") as fc:
        print(f[frame_path_suf].attrs['len'])
        for clip_idx, path_label in enumerate(fc.read().splitlines()):
            assert len(path_label.split()) == 2
            path,_= path_label.split()
            if(f[path].attrs['len']<1):
                # print(path)
                data=f[path][:4]
                print(data.shape)
                break
    f.close()
if __name__=="__main__":
    # test()
    # full_convert('/home/hzh/dfdc_1fps_crop/')
    # print(os.path.join('ma','gdsg','dfag'))
    # print(glob.glob('/home/hzh/dfdc_1fps_crop/method_A/1003254/1003254_A/1255229_1003254_A_001'+'/*.png'))
    read_test()