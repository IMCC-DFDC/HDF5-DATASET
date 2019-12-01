import torch
import sys
# sys.path.append('/home/ljm/SlowFast/')
import dataset_hdf5
import time
import numpy as np
# import torchvision.transforms as transforms
def main():
    dataset=dataset_hdf5.Dataset_hdf5("/home/ljm/gdata/DFDC/init/video/fb_dfd_release_0.1_final/three/",
                                      "/home/ljm/gdata/DFDC/init/frames/fps1_crop256.hdf5",
                                      'train')
    shuffle=True
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=shuffle,
        sampler=None,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    end = time.time()
    time_cost=[]
    for cur_iter, (inputs, labels, _) in enumerate(loader):
        time_cost.append(time.time() - end)
        print("dataload time {}".format(time.time() - end))
        end=time.time()
    print(f"{sum(time_cost):.3f}, "
          f"{np.mean(time_cost):.3f}, "
          f"{np.std(time_cost):.3f}, "
          f"{max(time_cost):.3f}, "
          f"{min(time_cost):.3f}")
if __name__ == '__main__':
    main()