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

class VecData:

    state_shape = (4, 14, 36)
    extra_shape = (4, 4)
    action_shape = (4, END)

    def __init__(self, link_shown, link_players, str2tile, tile2str, quan):
        self.str2tile = {}
        self.obs = np.zeros(self.state_shape, dtype=np.int32)
        self.extra = np.zeros(self.extra_shape, dtype=np.int32)
        self.link_shown = link_shown
        self.link_players = link_players
        self.str2tile = str2tile
        self.tile2str = tile2str
        self.mask = np.zeros(self.action_shape, dtype=np.bool)
        self.quan = quan
        

    def enable_pass(self):
        self.mask[:, PASS_MASK] = 1

    def enable_hu(self, player):
        self.mask[player, HU_MASK] = 1

    def check_able_play(self, player, canGANG=True):
        # hand[0] : self.obs[player, 4, :34]
        # hand[3] : self.obs[player, 7, :34]
        self.mask[player, PASS_MASK] = 0
        hand = self.obs[player, 4:8, :34].sum(0)
        self.mask[player][PLAY_MASK:CHI_MASK] = hand
        if canGANG and self.link_players[(player + 1) % 4].pTileWall != [] and self.link_players[player % 4].pTileWall != []:
            self.mask[player, ANGANG_MASK:HU_MASK] = hand == 4
            self.mask[player, BUGANG_MASK:ANGANG_MASK] = hand * self.obs[player, 12, :34]

    def check_able_ming(self, offer, tile):
        if self.link_players[(offer + 1) % 4].pTileWall == []:
            return
        canCHI = tile[0] in 'WBT'
        num = int(tile[1])
        tile = self.str2tile[tile]
        hand = self.obs[:, 4:8, :34].sum(1)

        for p in range(4):
            if p == offer:
                continue
            # check peng gang
            if hand[p, tile] >= 3:
                self.mask[p, GANG_MASK + tile] = 1
            if hand[p, tile] >= 2:
                self.mask[p, PENG_MASK + tile] = 1

        if canCHI:
            # check chi: (offer + 1) % 4
            p = (offer + 1) % 4
            hand[p, tile] += 1
            handp = hand[p].astype(np.bool)

            valid = handp[tile-num+1:tile-num+8] & handp[tile-num+2:tile-num+9] & handp[tile-num+3:tile-num+10]

            view = self.mask[p, CHI_MASK:PENG_MASK].reshape(3, 7, 3)[tile // 9]
            view[num-3:num-2, 2] = valid[num-3:num-2]
            view[num-2:num-1, 1] = valid[num-2:num-1]
            view[num-1:num, 0] = valid[num-1:num]

        
    def realize(self, action):
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

    def sync(self, shanten, idxs=[0, 1, 2, 3]):
        self.obs.fill(0)
        self.extra.fill(0)
        self.mask.fill(0)

        shownFlatten = np.array(tuple(i for i in self.link_shown.values()), dtype=np.int32)

        for a in idxs:
            shown = self.obs[a, 0:4, :34]
            hand = self.obs[a, 4:8, :34]
            chi_pack = self.obs[a, 8:12, :34]
            peng_pack = self.obs[a, 12, :34]
            gang_pack = self.obs[a, 13, :34]

            handFlatten = np.zeros_like(shownFlatten)
            chiFlatten = np.zeros_like(shownFlatten)
            for tile in self.link_players[a].tile:
                idx = self.str2tile[tile]
                handFlatten[idx] += 1
            
            for pack in self.link_players[a].pack:
                idx = self.str2tile[pack.tile]
                if pack.type == "CHI":
                    chiFlatten[idx] += 1
                elif pack.type == "PENG":
                    peng_pack[idx] = 1
                elif pack.type == "GANG":
                    gang_pack[idx] = 1
                else:
                    raise ValueError

            for i in range(1, 5):
                hand[0:i, handFlatten == i] = 1
                shown[0:i, (shownFlatten + handFlatten) == i] = 1
                chi_pack[0:i, chiFlatten == i] = 1

            self.extra[a, 0] = len(self.link_players[a].pTileWall)
            self.extra[a, 1] = a
            self.extra[a, 2] = self.quan
            self.extra[a, 3] = shanten[a]

    def get_obs(self, main=0, other=[1, 2, 3]):
        return ({
            'obs':{
                'obs': self.obs[main],
                'extra': self.extra[main]
            },
            'mask': self.mask[main]
        },
        {
            'obs':{
                'obs': self.obs[other],
                'extra': self.extra[other]
            },
            'mask': self.mask[other]
        })

# type: str, tile: str, offer: int
class Pack:
    def __init__(self, type, tile, offer):
        self.type = type
        self.tile = tile
        self.offer = offer
    def as_tuple(self):
        return self.type, self.tile, self.offer

# pack: list[Pack], tile: list[str], flower: list[flower], pTileWall: list[str]
class PlayerData:
    def __init__(self, pack, tile, flower, pTileWall):
        self.pack = pack
        self.tile = tile
        self.flower = flower
        self.pTileWall = pTileWall