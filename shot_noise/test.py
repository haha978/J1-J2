from utils import get_nearest_neighbors, flatten_neighbor_l

def main():
    print(flatten_neighbor_l(get_nearest_neighbors(3,3),3, 3))

if __name__ == '__main__':
    main()
