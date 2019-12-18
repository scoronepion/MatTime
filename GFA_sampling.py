import pickle
import torch
from torch.distributions.normal import Normal

if __name__ == '__main__':
    with open('/home/lab106/zy/MatTime/GFA_trans_nan.pk', 'rb') as f:
        raw = pickle.load(f)
    
    CRA = raw.loc[raw['Phase'] == 1]
    CRA_mean = CRA.mean()[:-4].fillna(0.0).values
    CRA_std = CRA.std()[:-4].fillna(0.0).values
    
    CRA_mean[7] = 0.0   # Pt
    CRA_mean[18] = 0.0  # Au
    CRA_mean[21] = 0.0  # Gd
    CRA_mean[22] = 0.0  # Rh
    CRA_mean[28] = 0.0  # Ta
    CRA_mean[30] = 0.0  # Dy
    CRA_mean[32] = 0.0  # Sm
    CRA_mean[33] = 0.0  # Pr
    CRA_mean[34] = 0.0  # Hf
    CRA_mean[35] = 0.0  # Ga
    CRA_mean[54] = 0.0  # Tb
    CRA_mean[55] = 0.0  # T

    CRA_std[7] = 0.0   # Pt
    CRA_std[18] = 0.0  # Au
    CRA_std[21] = 0.0  # Gd
    CRA_std[22] = 0.0  # Rh
    CRA_std[28] = 0.0  # Ta
    CRA_std[30] = 0.0  # Dy
    CRA_std[32] = 0.0  # Sm
    CRA_std[33] = 0.0  # Pr
    CRA_std[34] = 0.0  # Hf
    CRA_std[35] = 0.0  # Ga
    CRA_std[54] = 0.0  # Tb
    CRA_std[55] = 0.0  # T
    
    CRA_normal = Normal(torch.tensor(CRA_mean), torch.tensor(CRA_std))
    while True:
        sam = CRA_normal.sample()
        if sam.min().item() >= 0:
            print(sam)
            break
    # print(CRA.info())