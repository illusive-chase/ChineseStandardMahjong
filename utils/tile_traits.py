# -*- coding: utf-8 -*-

__all__ = ('Pack', 'PlayerData', 'str2tile', 'tile2str', 'tile_augment')

import numpy as np

str2tile = {}
tile2str = []
for k in "WBT":
    for i in range(1, 10):
        str2tile[k + str(i)] = len(tile2str)
        tile2str.append(k + str(i))
for i in range(1, 5):
    str2tile["F" + str(i)] = len(tile2str)
    tile2str.append("F" + str(i))
for i in range(1, 4):
    str2tile["J" + str(i)] = len(tile2str)
    tile2str.append("J" + str(i))

def tile_augment(tile, aug_type):
    if tile // 9 == 3:
        return tile
    if aug_type // 6 == 1:
        # 123456789 to 987654321
        tile = (tile // 9) * 9 + 8 - (tile % 9)
    table = [list(map(int, x)) for x in ['012', '021', '102', '120', '201', '210']]
    tile = table[aug_type % 6][tile // 9] * 9 + (tile % 9)
    return tile


# type: str, tile: str, offer: int
class Pack:
    def __init__(self, type, tile, offer, isANGANG=False):
        self.type = type
        self.tile = tile
        self.offer = offer
        self.isANGANG = isANGANG
    def as_tuple(self, player):
        return self.type, self.tile, (3 if self.type == 'CHI' else (4 + self.offer - player) % 4)

# pack: list[Pack], tile: list[str], flower: list[flower], pTileWall: list[str]
class PlayerData:
    def __init__(self, sync_info, pseudo):
        self.pack = []
        self.tile = []
        self.flower = []
        self.pTileWall = []
        self.sync = sync_info is not None
        self.pseudo = pseudo
        if self.sync:
            sync_array, idx = sync_info
            self.__hand = sync_array[idx, 4:8, :34]
            self.__chi = sync_array[idx, 8:12, :34]
            self.__peng = sync_array[idx, 24, :34]
            self.__gang = sync_array[idx, 28, :34]
            self.__angang = sync_array[idx, 32, :34]
            self.__history = (sync_array[idx, 33:61, :34], sync_array[(idx+3)%4, 61:89, :34], sync_array[(idx+2)%4, 89:117, :34], sync_array[(idx+1)%4, 117:145, :34])
            self.__round = 0
            self.__fhand = np.zeros(34, dtype=np.uint8)
            self.__fchi = np.zeros(34, dtype=np.uint8)
            global tile2str, str2tile
            self.tile2str = tile2str
            self.str2tile = str2tile

    def get_hand_count(self):
        return self.__fhand

    def draw(self, tile_str):
        self.tile.append(tile_str)
        if (not self.pseudo) and self.sync:
            tile_t = self.str2tile[tile_str]
            self.__hand[self.__fhand[tile_t], tile_t] = 1
            self.__fhand[tile_t] += 1
    
    def play(self, tile_str):
        if self.pseudo:
            self.tile.pop(-1)
            if self.sync:
                tile_t = self.str2tile[tile_str]
                for history in self.__history:
                    history[self.__round, tile_t] = 1
                self.__round += 1
            return True

        if tile_str not in self.tile:
            return False
        self.tile.remove(tile_str)
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__fhand[tile_t] -= 1
            self.__hand[self.__fhand[tile_t], tile_t] = 0
            for history in self.__history:
                history[self.__round, tile_t] = 1
            self.__round += 1
        return True

    def angang(self, tile_str, offer):
        if self.pseudo:
            self.tile = self.tile[:-4]
            return True
        if self.tile.count(tile_str) < 4:
            return False
        for _ in range(4):
            self.tile.remove(tile_str)
        self.pack.append(Pack("GANG", tile_str, offer, isANGANG=True))
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__angang[tile_t] = 1
            self.__fhand[tile_t] = 0
            self.__hand[:, tile_t] = 0
        return True

    def bugang(self, tile_str):
        if self.pseudo:
            self.tile.pop(-1)
            for i, pack in enumerate(self.pack):
                if pack.type == "PENG" and pack.tile == tile_str:
                    self.pack[i] = Pack("GANG", pack.tile, pack.offer)
                    if self.sync:
                        tile_t = self.str2tile[tile_str]
                        self.__gang[tile_t] = 1
                        self.__peng[tile_t] = 0
                    return True
            raise ValueError
        if tile_str not in self.tile:
            return False
        for i, pack in enumerate(self.pack):
            if pack.type == "PENG" and pack.tile == tile_str:
                self.pack[i] = Pack("GANG", pack.tile, pack.offer)
                self.tile.remove(tile_str)
                if self.sync:
                    tile_t = self.str2tile[tile_str]
                    self.__gang[tile_t] = 1
                    self.__peng[tile_t] = 0
                    self.__fhand[tile_t] -= 1
                    self.__hand[self.__fhand[tile_t], tile_t] = 0
                return True
        return False

    def gang(self, tile_str, offer):
        if self.pseudo:
            self.pack.append(Pack("GANG", tile_str, offer))
            self.tile = self.tile[:-3]
            if self.sync:
                tile_t = self.str2tile[tile_str]
                self.__gang[tile_t] = 1
            return True
        if self.tile.count(tile_str) < 3:
            return False
        self.pack.append(Pack("GANG", tile_str, offer))
        for _ in range(3):
            self.tile.remove(tile_str)
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__gang[tile_t] = 1
            self.__fhand[tile_t] -= 3
            self.__hand[self.__fhand[tile_t]:self.__fhand[tile_t]+3, tile_t] = 0
        return True

    def peng(self, tile_str, offer):
        if self.pseudo:
            self.pack.append(Pack("PENG", tile_str, offer))
            self.tile = self.tile[:-2]
            if self.sync:
                tile_t = self.str2tile[tile_str]
                self.__peng[tile_t] = 1
            return True
        if self.tile.count(tile_str) < 2:
            return False
        for _ in range(2):
            self.tile.remove(tile_str)
        self.pack.append(Pack("PENG", tile_str, offer))
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__peng[tile_t] = 1
            self.__fhand[tile_t] -= 2
            self.__hand[self.__fhand[tile_t]:self.__fhand[tile_t]+2, tile_t] = 0
        return True

    def chi(self, mid_tile_str, tile_str):
        c = mid_tile_str[0] + chr(ord(mid_tile_str[1]) - 1)
        for i in [-1, 0, 1]:
            if self.pseudo:
                self.tile.pop(-1)
            else:
                if c not in self.tile:
                    return False
                self.tile.remove(c)
            c = c[0] + chr(ord(c[1]) + 1)
        self.lastOp = "CHI"
        self.tileCHI = mid_tile_str
        self.pack.append(Pack("CHI", mid_tile_str, ord(tile_str[1]) - ord(mid_tile_str[1]) + 1))
        if self.sync:
            tile_t = self.str2tile[mid_tile_str]
            self.__chi[self.__fchi[tile_t], tile_t] = 1
            self.__fchi[tile_t] += 1
            if not self.pseudo:
                self.__fhand[tile_t-1:tile_t+2] -= 1
                self.__hand[self.__fhand[tile_t-1], tile_t-1] = 0
                self.__hand[self.__fhand[tile_t], tile_t] = 0
                self.__hand[self.__fhand[tile_t+1], tile_t+1] = 0
        return True