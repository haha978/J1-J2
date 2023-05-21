import numpy as np
import os

def main():
    input_dir = "ALA_3layers_tr5"
    #give your list here
    param_idx_l = list(range(0, 400, 80)) + list(range(400, 800, 20))
    np.save(os.path.join(input_dir, "param_idx_l.npy"), np.array(param_idx_l))


if __name__ == '__main__':
    main()
