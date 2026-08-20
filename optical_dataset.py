from __future__ import print_function, division

from torch.utils.data import Dataset


class OptDataSet(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image_dat = sample[0][0]
        wise_mag_info = sample[0][1]
        ps_ra, ps_dec = sample[0][2]

        if self.transform:
            image_dat = self.transform(image_dat)

        label = sample[1]
        return image_dat, label, wise_mag_info, (ps_ra, ps_dec)
