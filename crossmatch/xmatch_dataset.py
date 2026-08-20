from __future__ import print_function, division

from torch.utils.data import Dataset


class XMatchDataset(Dataset):
    def __init__(self, dataset, radio_transform=None, opt_transform=None):
        self.dataset = dataset
        self.radio_transforms = radio_transform
        self.opt_transforms = opt_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):  # radio_img_dat, ps_bg_imgcube, ps_imgcube, wise_magnitude_info, cutout_position_info, label
        sample = self.dataset[idx]
        radio_img_data = sample[0]
        ps_sample_imgcube = sample[1]
        wise_mag_info = sample[2]
        ps_cutout_pos_info = sample[3]
        centroid = sample[4]
        geodesic = sample[5]
        label = sample[6]
        ps_source_identity = sample[7]
        VLASS_name = sample[8]

        if self.radio_transforms:
            radio_img_data = self.radio_transforms(radio_img_data)

        if self.opt_transforms:
            ps_sample_imgcube = self.opt_transforms(ps_sample_imgcube)

        return radio_img_data, ps_sample_imgcube, wise_mag_info, ps_cutout_pos_info, centroid, geodesic, label, ps_source_identity, VLASS_name
