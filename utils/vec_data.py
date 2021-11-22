# -*- coding: utf-8 -*-

__all__ = ('VecData',)


import numpy as np
from utils.tile_traits import PlayerData, str2tile, tile2str

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


class VecData:

    state_shape = (4, 145, 36)
    action_shape = (4, END)

    def __init__(self, quan, no_obs_players=[]):
        self.obs = np.zeros(self.state_shape, dtype=np.bool)
        self.str2tile = str2tile
        self.tile2str = tile2str
        self.mask = np.zeros(self.action_shape, dtype=np.bool)
        self.players = [PlayerData((self.obs, i), (i in no_obs_players)) for i in range(4)]
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
        hand = self.players[player].get_hand_count()
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

        
