#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
from io import BytesIO
import torch
import torch.utils.data
import h5py
# import slowfast.datasets.transform as transform



class Dataset_hdf5(torch.utils.data.Dataset):
    """
    hdf5 dataset loader. Construct the hdf5 dataset loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, datalistdir,datafile, mode, sample_rate=4,sample_number=4,
                        num_retries=10,clip_idx=0,num_ensemble_views=1,
                        transform=None):
        """
        Construct the Dataset_hdf5 loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            datalistdir(str）：the dir of the given csv file
            datafile(str):the path to the HDF5 File
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            sample_rate(int):sampling rate
            sample_number(int):The number of frames
            num_retries (int): number of retries.
            clip_idx(int): For the test mode, this parameter is used to specify which chip is selected
            num_ensemble_views(int)：For the test mode, the video will divide into num_ensemble_views clips.
            transform():data augmentation
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Kinetics".format(mode)
        self.datalistdir=datalistdir
        self.datafile=datafile
        self.mode = mode
        # self.cfg = cfg
        self.sample_rate=sample_rate
        self.sample_num=sample_number
        self._num_retries = num_retries
        self.transform=transform
        # self.clip_idx=clip_idx
        # self.NUM_ENSEMBLE_VIEWS=NUM_ENSEMBLE_VIEWS
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            assert clip_idx<NUM_ENSEMBLE_VIEWS,'clip_idx Incompatible with numensemble_views'
            self._num_clips = (
                num_ensemble_views
            )
            self.clip_idx = clip_idx
        print("Constructing hdf5 dataset {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.datalistdir, "{}.csv".format(self.mode)
        )
        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        with open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert len(path_label.split()) == 2
                path, label = path_label.split()
                # for idx in range(self._num_clips):
                self._path_to_videos.append(
                    path
                )
                self._labels.append(int(label))
                    # self._spatial_temporal_idx.append(idx)
                    # self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load datafile split {} from {}".format(
            self.mode, path_to_file
        )
        # self.f = h5py.File(self.datafile, "r")
        # assert (self.f != None), "Failed to open dataset file {}".format(self.datafile)
        print(
            "Constructing hdf5 dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )


    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index =self.clip_idx

        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for _ in range(self._num_retries):
            frames=self.h5decode(self._path_to_videos[index],
                                 self.sample_rate,
                                 self.sample_num,
                                 temporal_sample_index,
                                 self._num_clips)
            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames is None:
                index = random.randint(0, len(self._path_to_videos) - 1)
                continue

            # Perform color normalization.
            frames = frames.float()
            frames = frames / 255.0
            # frames = frames - torch.tensor(self.cfg.DATA.MEAN)
            # frames = frames / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            # frames = frames.permute(3, 0, 1, 2)
            # Perform data augmentation.
            if self.transform!=None:
                frames=self.transform(frames)
            label = self._labels[index]
            return frames, label, index
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)


    def h5decode(self,path,
                 sampling_rate,
                 num_frames,
                 clip_idx=-1,
                 num_clips=10):
        """
            Decode the HDF5 file and perform temporal sampling.
            Args:
                path(str):the path to frames in the HDF5 file
                sampling_rate (int): frame sampling rate (interval between two sampled
                    frames).
                num_frames (int): number of frames to sample.
                clip_idx (int): if clip_idx is -1, perform random temporal
                    sampling. If clip_idx is larger than -1, uniformly split the
                    video to num_clips clips, and select the
                    clip_idx-th video clip.
                num_clips (int): overall number of clips to uniformly
                    sample from the given video.
            Returns:
                frames (tensor): decoded frames from the video.
            """
        assert clip_idx >= -1, "Not valied clip_idx {}".format(clip_idx)
        try:
            with h5py.File(self.datafile,'r') as f:
                start_idx, end_idx = get_start_end_idx(
                    f[path].attrs['len'],
                    sampling_rate * num_frames,
                    clip_idx,
                    num_clips
                )
                frames=f[path][start_idx:end_idx+1]
            frames = torch.as_tensor(frames)
        except Exception as e:
            print("Failed to get frames: {}".format(e))
            return None
        if frames.shape[0]==0:
            return None
        frames = temporal_sampling(frames, 0, frames.shape[0]-1, num_frames)
        return frames

def get_start_end_idx(video_size, clip_size, clip_idx, num_clips):
    """
    Sample a clip of size clip_size from a video of size video_size and
    return the indices of the first and last frame of the clip. If clip_idx is
    -1, the clip is randomly sampled, otherwise uniformly split the video to
    num_clips clips, and select the start and end index of clip_idx-th video
    clip.
    Args:
        video_size (int): number of overall frames.
        clip_size (int): size of the clip to sample from the frames.
        clip_idx (int): if clip_idx is -1, perform random jitter sampling. If
            clip_idx is larger than -1, uniformly split the video to num_clips
            clips, and select the start and end index of the clip_idx-th video
            clip.
        num_clips (int): overall number of clips to uniformly sample from the
            given video for testing.
    Returns:
        start_idx (int): the start frame index.
        end_idx (int): the end frame index.
    """
    delta = max(video_size - clip_size, 0)
    if clip_idx == -1:
        # Random temporal sampling.
        start_idx = random.randint(0, delta)
    else:
        # Uniformly sample the clip with the given index.
        start_idx = delta * clip_idx / num_clips
    end_idx = start_idx + clip_size - 1
    return int(start_idx), int(end_idx)
def temporal_sampling(frames, start_idx, end_idx, num_samples):
    """
    Given the start and end frame index, sample num_samples frames between
    the start and end with equal interval.
    Args:
        frames (tensor): a tensor of video frames, dimension is
            `num video frames` x `channel` x `height` x `width`.
        start_idx (int): the index of the start frame.
        end_idx (int): the index of the end frame.
        num_samples (int): number of frames to sample.
    Returns:
        frames (tersor): a tensor of temporal sampled video frames, dimension is
            `num clip frames` x `channel` x `height` x `width`.
    """
    index = torch.linspace(start_idx, end_idx, num_samples)
    index = torch.clamp(index, 0, frames.shape[0] - 1).long()
    frames = torch.index_select(frames, 0, index)
    return frames