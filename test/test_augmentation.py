import argparse
import tqdm
from utils.paired_data import PairedDataset
from utils.vec_data import VecData
from utils.tile_traits import tile_augment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pair_file', type=str)
    args = parser.parse_args()
    pdataset = PairedDataset()
    with tqdm.trange(10000, desc=f"Checking VecData", dynamic_ncols=True, ascii=True) as t:
        for i in t:
            action = i % 235
            tile = (i // 235) % 34
            for j in range(12):
                augged_action = action
                augged_tile = tile
                augged_tile2 = tile
                for k in range(6):
                    augged_action = VecData.action_augment_table[j, augged_action]
                    augged_tile = VecData.tile_augment_table[j, augged_tile]
                    augged_tile2 = tile_augment(augged_tile2, j)
                assert augged_tile2 == tile, (j, tile, augged_tile2)
                assert augged_action == action, (j, action, augged_action)
                assert augged_tile == tile, (j, tile, augged_tile)

    with open(args.pair_file, 'rb') as f:
        pdataset.load(f, 10000)
    with tqdm.trange(10000, desc=f"Checking PairedDataset", dynamic_ncols=True, ascii=True) as t:
        for i in t:
            for j in range(12):
                augged = pdataset.get(i)
                for k in range(6):
                    augged = PairedDataset.augment(augged, j)
                equal = bool((augged == pdataset.get(i)).all())
                assert equal, (j, (augged != pdataset.get(i)).nonzero()[0])