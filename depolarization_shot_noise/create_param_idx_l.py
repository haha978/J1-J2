import numpy as np
import os

def main():
    input_dir = "3x2_J1_J2_tr2_sim"
    #give your list here
    param_idx_l = list(range(0, 150, 30)) + list(range(150, 251, 25)) + list(range(300, 480, 6))
    np.save(os.path.join(input_dir, "param_idx_l.npy"), np.array(param_idx_l))


if __name__ == '__main__':
    main()
