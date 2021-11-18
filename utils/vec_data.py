# -*- coding: utf-8 -*-

__all__ = ('VecData', 'Pack', 'PlayerData')


import numpy as np

PLAY_MASK = 0
CHI_MASK = 34
PENG_MASK = CHI_MASK+63
GANG_MASK = PENG_MASK+34
BUGANG_MASK = GANG_MASK+34
ANGANG_MASK = BUGANG_MASK+34
HU_MASK = ANGANG_MASK+34
PASS_MASK = HU_MASK+1
END = PASS_MASK+1
assert END == 235


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

class VecData:

    state_shape = (4, 145, 36)
    action_shape = (4, END)

    def __init__(self, quan, no_obs_players):
        global str2tile, tile2str
        self.obs = np.zeros(self.state_shape, dtype=np.bool)
        self.str2tile = str2tile
        self.tile2str = tile2str
        self.mask = np.zeros(self.action_shape, dtype=np.bool)
        self.players = [PlayerData(None if i in no_obs_players else self.obs[i]) for i in range(4)]
        self.fshown = np.zeros(34, dtype=np.uint8)
        self.shown = np.zeros((4, 34), dtype=np.bool)
        self.no_obs_players = no_obs_players
        self.quan = quan

    def connect(self):
        return self.players, self.fshown, self.str2tile, self.tile2str

    def enable_pass(self):
        self.mask[:, PASS_MASK] = 1

    def enable_hu(self, player):
        self.mask[player, HU_MASK] = 1

    def check_able_play(self, player, canGANG=True):
        # hand[0] : self.obs[player, 4, :34]
        # hand[3] : self.obs[player, 7, :34]
        if player in self.no_obs_players:
            return
        self.mask[player, PASS_MASK] = 0
        hand = self.players[player].fhand
        self.mask[player][PLAY_MASK:CHI_MASK] = hand
        if canGANG and self.players[(player + 1) % 4].pTileWall != [] and self.players[player % 4].pTileWall != []:
            self.mask[player, ANGANG_MASK:HU_MASK] = hand == 4
            self.mask[player, BUGANG_MASK:ANGANG_MASK] = hand & self.obs[player, 24, :34]

    def check_able_ming(self, offer, tile):
        if self.players[(offer + 1) % 4].pTileWall == []:
            return
        canCHI = tile[0] in 'WBT'
        num = int(tile[1])
        tile = self.str2tile[tile]
        hand = self.obs[:, 4:8, :34]

        for p in range(4):
            if p == offer:
                continue
            # check peng gang
            if hand[p, 2, tile]:
                self.mask[p, GANG_MASK + tile] = 1
            if hand[p, 1, tile]:
                self.mask[p, PENG_MASK + tile] = 1

        if canCHI:
            # check chi: (offer + 1) % 4
            p = (offer + 1) % 4
            handp = hand[p, 0, :].copy()
            handp[tile] = True

            valid = handp[tile-num+1:tile-num+8] & handp[tile-num+2:tile-num+9] & handp[tile-num+3:tile-num+10]

            if valid[max(0, num-3):num].any():
                view = self.mask[p, CHI_MASK:PENG_MASK].reshape(3, 7, 3)[tile // 9]
                if 2 < num:
                    view[num-3, 2] = valid[num-3]
                if 1 < num < 9:
                    view[num-2, 1] = valid[num-2]
                if num < 8:
                    view[num-1, 0] = valid[num-1]

    def translate(self, action, serve=None):
        lst = action.split()
        if lst[0] == "PASS":
            return PASS_MASK
        if lst[0] == "PLAY":
            return PLAY_MASK + self.str2tile[lst[1]]
        if lst[0] == "CHI":
            idx = 'WBT'.index(lst[1][0]) * 7 + ord(lst[1][1]) - 50
            offset = ord(serve[1]) - ord(lst[1][1]) + 1
            return CHI_MASK + offset + idx * 3
        if lst[0] == "PENG":
            return PENG_MASK + self.str2tile[serve]
        if lst[0] == "GANG":
            if len(lst) == 1:
                return GANG_MASK + self.str2tile[serve]
            return ANGANG_MASK + self.str2tile[lst[1]]
        if lst[0] == "BUGANG":
            return BUGANG_MASK + self.str2tile[lst[1]]
        if lst[0] == "HU":
            return HU_MASK
        raise ValueError
        
    def realize(self, action):
        if type(action) is str:
            return action
        if action >= PASS_MASK:
            return "PASS"
        if action < CHI_MASK:
            return "PLAY " + self.tile2str[action - PLAY_MASK]
        if action < PENG_MASK:
            idx = (action - CHI_MASK) // 3
            idx = (idx // 7) * 9 + (idx % 7) + 1
            return "CHI " + self.tile2str[idx] + " ??"
        if action < GANG_MASK:
            return "PENG ??"
        if action < BUGANG_MASK:
            return "GANG"
        if action < ANGANG_MASK:
            return "BUGANG " + self.tile2str[action - BUGANG_MASK]
        if action < HU_MASK:
            return "GANG " + self.tile2str[action - ANGANG_MASK]
        if action < PASS_MASK:
            return "HU"
        raise ValueError

    def sync(self, idxs=[0, 1, 2, 3]):
        self.mask.fill(0)

        # cross-copy

        chi_pack = self.obs[:, 8:24, :].reshape(4, 4, 4 * 36)
        peng_pack = self.obs[:, 24:28, :].reshape(4, 4, 36)
        gang_pack = self.obs[:, 28:32, :].reshape(4, 4, 36)

        for a in idxs:
            if a in self.no_obs_players:
                continue
            chi_pack[a, 1] = chi_pack[(a + 1) % 4, 0]
            chi_pack[a, 2] = chi_pack[(a + 2) % 4, 0]
            chi_pack[a, 3] = chi_pack[(a + 3) % 4, 0]
            peng_pack[a, 1] = peng_pack[(a + 1) % 4, 0]
            peng_pack[a, 2] = peng_pack[(a + 2) % 4, 0]
            peng_pack[a, 3] = peng_pack[(a + 3) % 4, 0]
            gang_pack[a, 1] = gang_pack[(a + 1) % 4, 0]
            gang_pack[a, 2] = gang_pack[(a + 2) % 4, 0]
            gang_pack[a, 3] = gang_pack[(a + 3) % 4, 0]

    def show(self, tile_str, num):
        tile_t = str2tile[tile_str]
        self.shown[self.fshown[tile_t]:self.fshown[tile_t]+num, tile_t] = 1
        self.fshown[tile_t] += num

    def show_range(self, begin, end, num):
        for tile_t in range(begin, end):
            self.shown[self.fshown[tile_t]:self.fshown[tile_t]+num, tile_t] = 1
        self.fshown[begin:end] += num

    def get_obs(self, main, other=[1, 2, 3]):
        return (
            (self.obs[main], self.mask[main], np.asarray(main)),
            (self.obs[other], self.mask[other], np.asarray(other))
        )
            

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
    def __init__(self, sync_array):
        self.pack = []
        self.tile = []
        self.flower = []
        self.pTileWall = []
        self.sync = sync_array is not None
        if self.sync:
            self.__hand = sync_array[4:8, :34]
            self.__chi = sync_array[8:12, :34]
            self.__peng = sync_array[24, :34]
            self.__gang = sync_array[28, :34]
            self.__angang = sync_array[32, :34]
            self.fhand = np.zeros(34, dtype=np.uint8)
            self.__fchi = np.zeros(34, dtype=np.uint8)
            global tile2str, str2tile
            self.tile2str = tile2str
            self.str2tile = str2tile

    def draw(self, tile_str):
        self.tile.append(tile_str)
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__hand[self.fhand[tile_t], tile_t] = 1
            self.fhand[tile_t] += 1
    
    def play(self, tile_str):
        if tile_str not in self.tile:
            return False
        self.tile.remove(tile_str)
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.fhand[tile_t] -= 1
            self.__hand[self.fhand[tile_t], tile_t] = 0
        return True

    def angang(self, tile_str, offer):
        if self.tile.count(tile_str) < 4:
            return False
        for _ in range(4):
            self.tile.remove(tile_str)
        self.pack.append(Pack("GANG", tile_str, offer, isANGANG=True))
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__angang[tile_t] = 1
            self.fhand[tile_t] = 0
            self.__hand[:, tile_t] = 0
        return True

    def bugang(self, tile_str):
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
                    self.fhand[tile_t] -= 1
                    self.__hand[self.fhand[tile_t], tile_t] = 0
                return True
        return False

    def gang(self, tile_str, offer):
        if self.tile.count(tile_str) < 3:
            return False
        self.pack.append(Pack("GANG", tile_str, offer))
        for _ in range(3):
            self.tile.remove(tile_str)
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__gang[tile_t] = 1
            self.fhand[tile_t] -= 3
            self.__hand[self.fhand[tile_t]:self.fhand[tile_t]+3, tile_t] = 0
        return True

    def peng(self, tile_str, offer):
        if self.tile.count(tile_str) < 2:
            return False
        for _ in range(2):
            self.tile.remove(tile_str)
        self.pack.append(Pack("PENG", tile_str, offer))
        if self.sync:
            tile_t = self.str2tile[tile_str]
            self.__peng[tile_t] = 1
            self.fhand[tile_t] -= 2
            self.__hand[self.fhand[tile_t]:self.fhand[tile_t]+2, tile_t] = 0
        return True

    def chi(self, mid_tile_str, tile_str):
        c = mid_tile_str[0] + chr(ord(mid_tile_str[1]) - 1)
        for i in [-1, 0, 1]:
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
            self.fhand[tile_t-1:tile_t+2] -= 1
            self.__hand[self.fhand[tile_t-1], tile_t-1] = 0
            self.__hand[self.fhand[tile_t], tile_t] = 0
            self.__hand[self.fhand[tile_t+1], tile_t+1] = 0
        return True
        
