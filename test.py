# -*- coding: utf-8 -*-
import torch
from torch import nn
import numpy as np
from MahjongGB import MahjongFanCalculator, MahjongShanten
import json

PLAY_MASK = 0
CHI_MASK = 34
PENG_MASK = CHI_MASK+34*34
GANG_MASK = PENG_MASK+34*34
BUGANG_MASK = GANG_MASK+34
ANGANG_MASK = BUGANG_MASK+34
HU_MASK = ANGANG_MASK+34
PASS_MASK = HU_MASK+1
END = PASS_MASK+1

class VecData:

    state_shape = (4, 14, 34)
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

    def check_able_play(self, player, tile):
        # hand[0] : self.obs[player, 4, :]
        # hand[3] : self.obs[player, 7, :]
        self.mask[player, PASS_MASK] = 0
        tile = self.str2tile[tile]
        hand = self.obs[player, 4:8, :].sum(0)
        hand[tile] += 1
        self.mask[player][PLAY_MASK:CHI_MASK] = hand
        if self.link_players[(player + 1) % 4].pTileWall != [] and self.link_players[player % 4].pTileWall != []:
            self.mask[player, ANGANG_MASK:HU_MASK] = hand == 4
            self.mask[player, BUGANG_MASK:ANGANG_MASK] = hand * self.obs[player, 12, :]

    def check_able_ming(self, offer, tile):
        if self.link_players[(offer + 1) % 4].pTileWall == []:
            return
        canCHI = tile[0] in 'WBT'
        num = int(tile[1])
        tile = self.str2tile[tile]
        hand = self.obs[:, 4:8, :].sum(1)

        for p in range(4):
            if p == offer:
                continue
            # check peng gang
            if hand[p, tile] >= 3:
                self.mask[p, GANG_MASK + tile] = 1
            if hand[p, tile] >= 2:
                hand_copy = np.copy(hand[p, :])
                hand_copy[tile] -= 2
                self.mask[p, PENG_MASK:GANG_MASK].reshape(34, 34)[tile, hand_copy.astype(np.bool)] = 1

        if canCHI:
            # check chi: (offer + 1) % 4
            p = (offer + 1) % 4
            hand[p, tile] += 1
            handp = hand[p].astype(np.bool)
            valid = np.zeros((34,), dtype=np.bool)
            valid[tile-num+2:tile-num+9] = handp[tile-num+1:tile-num+8] & handp[tile-num+2:tile-num+9] & handp[tile-num+3:tile-num+10]
            valid[:tile-1] = False
            valid[tile+2:] = False
            view = self.mask[p, CHI_MASK:PENG_MASK].reshape(34, 34)
            view[np.ix_(valid, handp)] = True
            for i in range(tile - 1, tile + 2):
                if i >= 0 and valid[i]:
                    view[i, i-1:i+2][hand[p, i-1:i+2] == 1] = False
        
    def realize(self, action):
        if action >= PASS_MASK:
            return "PASS"
        if action < CHI_MASK:
            return "PLAY " + self.tile2str[action]
        if action < PENG_MASK:
            action = action - CHI_MASK
            return "CHI " + self.tile2str[action // 34] + " " + self.tile2str[action % 34]
        if action < GANG_MASK:
            action = action - PENG_MASK
            return "PENG " + self.tile2str[action % 34]
        if action < BUGANG_MASK:
            return "GANG"
        if action < ANGANG_MASK:
            action = action - BUGANG_MASK
            return "BUGANG " + self.tile2str[action % 34]
        if action < HU_MASK:
            action = action - ANGANG_MASK
            return "GANG " + self.tile2str[action % 34]
        if action < PASS_MASK:
            return "HU"
        raise ValueError

    def sync(self, shanten, idxs=[0, 1, 2, 3]):
        self.obs.fill(0)
        self.extra.fill(0)
        self.mask.fill(0)

        shownFlatten = np.array(tuple(i for i in self.link_shown.values()), dtype=np.int32)

        for a in idxs:
            shown = self.obs[a, 0:4, :]
            hand = self.obs[a, 4:8, :]
            chi_pack = self.obs[a, 8:12, :]
            peng_pack = self.obs[a, 12, :]
            gang_pack = self.obs[a, 13, :]

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


class Net(nn.Module):
    def __init__(self, state_shape, extra_shape, action_shape, device):
        super().__init__()
        self.embedding1 = nn.Embedding(22, 8)
        self.embedding2 = nn.Embedding(4, 8)
        self.embedding3 = nn.Embedding(8, 8)
        self.linear = nn.Sequential(*[
            nn.Linear(np.prod(state_shape[1:]), 256), nn.ReLU(inplace=True),
        ])
        self.model = nn.Sequential(*[
            nn.Linear(256+8*4, 256), nn.ReLU(inplace=True)
        ])
        self.device = device
    def forward(self, s, **kwargs):
        obs = torch.as_tensor(s['obs']['obs'], device=self.device, dtype=torch.float32)
        extra = torch.as_tensor(s['obs']['extra'], device=self.device, dtype=torch.int32)
        batch = obs.shape[0]
        state = self.linear(obs.view(batch, -1))
        return self.model(torch.cat((
            state,
            self.embedding1(extra[:, 0]),
            self.embedding2(extra[:, 1]),
            self.embedding2(extra[:, 2]),
            self.embedding3(extra[:, 3])
        ), dim=1))

class Actor(nn.Module):
    def __init__(self, net, state_shape, extra_shape, action_shape, device):
        super().__init__()
        self.net = net
        self.model = nn.Sequential(*[
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, np.prod(action_shape[1:]), bias=False)
        ])
        self.device = device
    def forward(self, s, state=None, info={}):
        mask = torch.as_tensor(s['mask'], device=self.device, dtype=torch.int32)
        probs = nn.Softmax(dim=1)(self.model(self.net(s))) * mask
        return probs, state


class Bot:

    state_shape = VecData.state_shape
    extra_shape = VecData.extra_shape
    action_shape = VecData.action_shape

    def __init__(self):
        pass
        
    def get_obs(self, requests, responses):

        _, id, quan = requests[0].split(' ')
        requests = requests[1:]

        # playerData: list[PlayerData]
        self.playerData = [None] * 4
        for i in range(4):
            self.playerData[i] = PlayerData([], [], [], [])
        self.lastTile = ''
        self.lastOp = ''

        # "PASS"
        # "PLAY"
        # "PENG"
        # "GANG"
        # "CHI"
        # "HU"

        self.tileCHI = ''
        self.lastBUGANG = False
        self.currBUGANG = False
        self.lastGANG = False
        self.currGANG = False
        self.lastANGANG = False
        self.currANGANG = False

        # roundStage: int, lastRoundStage: int
        self.roundStage = -2
        self.lastRoundStage = None

        # -2:通知位置
        # -1:发牌
        # 0-3:玩家摸牌
        # 4-7:玩家打出牌后，通知所有玩家
        # 8-12:玩家杠牌，通知所有玩家

        # quan: int
        self.quan = int(quan)
        self.id = int(id)
        # tileWall: list[str]
        # self.shownTile: dict[str:int]
        self.shownTile = {}
        self.str2tile = {}

        for k in "WBT":
            for i in range(1, 10):
                self.shownTile[k + str(i)] = 0
        for i in range(1, 5):
            self.shownTile["F" + str(i)] = 0
        for i in range(1, 4):
            self.shownTile["J" + str(i)] = 0

        self.tile2str = list(self.shownTile.keys())
        for idx, tile in enumerate(self.tile2str):
            self.str2tile[tile] = idx

        self.vec_data = VecData(self.shownTile, self.playerData, self.str2tile, self.tile2str, self.quan)
        self.canHu = 0
        
        

        partLength = 136 // 4
        for i in range(4):
            for j in range(partLength):
                self.playerData[i].pTileWall.append('??')

        for response, request in zip(responses, requests):
            inferred_response = self.infer_response(request)
            inferred_response[self.id] = response
            self.roundInput(inferred_response)
            self.roundOutput(request)


        # out->request->response->input->out->request
        #                  ^              ^      |
        #                  |              |      |
        #                  -----------------------

        return self.vec_data.get_obs([self.id])[0]


    def playerError(self, player: int, code: str):
        raise ValueError

    def checkHu(self, player: int, re: int):
        if re == -1:
            try:
                fan_table = MahjongFanCalculator(
                    hand = tuple(self.playerData[player].tile),
                    pack = tuple(v.as_tuple() for v in self.playerData[player].pack),
                    winTile = self.lastTile,
                    flowerCount = len(self.playerData[player].flower),
                    isSelfDrawn = self.roundStage == player,
                    is4thTile = self.shownTile[self.lastTile] == 3,
                    isAboutKong = self.lastBUGANG or self.lastANGANG or self.currGANG,
                    isWallLast = self.playerData[(self.roundStage + 1) % 4].pTileWall == [],
                    seatWind = player,
                    prevalentWind  = self.quan,
                    verbose = True
                )
            except TypeError as e:
                assert "ERROR_NOT_WIN" == str(e)
                fan_table = ()
            finally:
                pass
            re = sum((t[0] for t in fan_table))
            if re < 8 + len(self.playerData[player].flower):
                re = 0
            return re
                
        if re < 8 + len(self.playerData[player].flower):
            self.playerError(player, "诈胡")
        if self.verbose:
            try:
                fan_table = MahjongFanCalculator(
                    hand = tuple(self.playerData[player].tile),
                    pack = tuple(p.as_tuple() for p in self.playerData[player].pack),
                    winTile = self.lastTile,
                    flowerCount = len(self.playerData[player].flower),
                    isSelfDrawn = self.roundStage == player,
                    is4thTile = self.shownTile[self.lastTile] == 3,
                    isAboutKong = self.lastBUGANG or self.lastANGANG or self.currGANG,
                    isWallLast = self.playerData[(self.roundStage + 1) % 4].pTileWall == [],
                    seatWind = player,
                    prevalentWind  = self.quan,
                    verbose = True
                )
            except TypeError as e:
                assert "ERROR_NOT_WIN" == str(e)
                fan_table = ()
            finally:
                pass
            self.display.append("结束：")
            self.display.append("胡牌：PLAYER {}".format(player))
            for fan_point, fan_count, fan_name, _ in fan_table:
                self.display.append("\t{} x {}：{} 番\t".format(fan_name, fan_count, fan_point))
            self.display.append("\t共：{} 番".format(re))
            for i in range(4):
                if self.roundStage < 4:
                    if i == player:
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, 3 * (8 + re)))
                    else:
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, -(8 + re)))
                else:
                    if i == player:
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, (3 * 8 + re)))
                    elif self.roundStage == i + 4:
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, -(8 + re)))
                    elif self.roundStage == i + 8 and (self.lastBUGANG or self.lastANGANG):
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, -(8 + re)))
                    else:
                        self.display.append("\tPLAYER {} SCORE {:+}".format(i, -8))

        for i in range(4):
            if self.roundStage < 4:
                if i == player:
                    self.rew[i] = 3 * (8 + re)
                else:
                    self.rew[i] = -(8 + re)
            else:
                if i == player:
                    self.rew[i] = 3 * (8 + re)
                elif self.roundStage == i + 4:
                    self.rew[i] = -(8 + re)
                elif self.roundStage == i + 8 and (self.lastBUGANG or self.lastANGANG):
                    self.rew[i] = -(8 + re)
                else:
                    self.rew[i] = -8
        raise FinishError
        return re


    def checkInputPASS(self, response: str, player: int):
        if response != "PASS":
            self.playerError(player, "未确认")


    def checkInputDRAW(self, response: str, player: int):
        outputList = response.split(' ')

        if len(outputList) == 1:
            if outputList[0] == "HU":
                self.checkHu(player, self.canHu[player])
        elif len(outputList) == 2:
            self.playerData[player].tile.append(self.lastTile)
            self.lastTile = outputList[1]
            if outputList[0] == "PLAY":
                if player != self.id:
                    self.playerData[player].tile.pop(-1)
                    self.lastOp = "PLAY"
                    self.roundStage += 4
                    return
                if self.lastTile in self.playerData[player].tile:
                    self.playerData[player].tile.remove(self.lastTile)
                    self.lastOp = "PLAY"
                    self.roundStage += 4
                    return
            elif outputList[0] == "GANG":
                if self.playerData[player].pTileWall == [] or self.playerData[(player + 1) % 4].pTileWall == []:
                    self.playerError(player, "终局杠牌")
                if player != self.id:
                    for i in range(4):
                        self.playerData[player].tile.pop(-1)
                    self.playerData[player].pack.append(Pack("GANG", '??', player))
                else:
                    for i in range(4):
                        if self.lastTile not in self.playerData[player].tile:
                            self.playerError(player, "无牌暗杠")
                        self.playerData[player].tile.remove(self.lastTile)
                    self.playerData[player].pack.append(Pack("GANG", self.lastTile, player))
                self.lastOp = "GANG"
                self.currANGANG = True
                self.currGANG = False
                self.lastGANG = False
                self.currBUGANG = False
                self.lastBUGANG = False
                self.roundStage = player + 8
                return
            elif outputList[0] == "BUGANG":
                for i, pack in enumerate(self.playerData[player].pack):
                    if pack.type == "PENG" and pack.tile == self.lastTile:
                        self.playerData[player].pack[i] = Pack("GANG", pack.tile, pack.offer)
                        if self.lastTile not in self.playerData[player].tile:
                            self.playerError(player, "无补杠对应刻子")
                        self.playerData[player].tile.remove(self.lastTile)
                        self.shownTile[self.lastTile] = 4
                        self.lastOp = "BUGANG"
                        self.currBUGANG = True
                        self.currANGANG = False
                        self.currGANG = False
                        self.lastGANG = False
                        self.lastBUGANG = False
                        self.roundStage = player + 8
                        return
        self.playerError(player, "非法操作 " + response)
                
            

    def checkInputPLAY1(self, response: str, player: int):
        if response == "HU":
            self.checkHu(player, self.canHu[player])

    def checkInputPLAY2(self, response: str, player: int):
        outputList = response.split(' ')
        if response == "PASS":
            return False
        elif response == "GANG":
            if player != self.id:
                for i in range(3):
                    self.playerData[player].tile.pop(-1)
            else:
                for i in range(3):
                    if self.lastTile not in self.playerData[player].tile:
                        self.playerError(player, "无刻子杠牌")
                    self.playerData[player].tile.remove(self.lastTile)
            self.shownTile[self.lastTile] = 4
            self.lastOp = "GANG"
            self.currGANG = True
            self.currBUGANG = False
            self.currANGANG = False
            self.lastGANG = False
            self.lastBUGANG = False
            self.playerData[player].pack.append(Pack("GANG", self.lastTile, self.roundStage % 4))
            self.roundStage = player + 8
            return True
        elif len(outputList) == 2:
            if outputList[0] == "PENG":
                if player != self.id:
                    for i in range(3):
                        self.playerData[player].tile.pop(-1)
                else:
                    for i in range(2):
                        if self.lastTile not in self.playerData[player].tile:
                            self.playerError(player, "无对子碰牌")
                        self.playerData[player].tile.remove(self.lastTile)
                self.shownTile[self.lastTile] += 3
                self.lastOp = "PENG"
                self.playerData[player].pack.append(Pack("PENG", self.lastTile, self.roundStage % 4))
                self.lastTile = outputList[1]
                if player != self.id:
                    self.playerData[player].tile.pop(-1)
                else:
                    if self.lastTile not in self.playerData[player].tile:
                        self.playerError(player, "打出非手牌 " + self.lastTile)
                    self.playerData[player].tile.remove(self.lastTile)
                self.roundStage = 4 + player
                return True
        if len(outputList) != 3:
            self.playerError(player, "非法操作 " + response)
        return False

    def checkInputPLAY3(self, response: str,  player: int):
        outputList = response.split(' ')
        if len(outputList) == 3:
            if outputList[0] != "CHI" or (self.roundStage - player) % 4 != 3:
                self.playerError(player, "非法操作 " + response)
            if player != self.id:
                self.playerData[player].tile.pop(-1)
                self.playerData[player].tile.pop(-1)
                c = outputList[1]
                c = c[0] + chr(ord(c[1]) - 1)
                for i in [-1, 0, 1]:
                    self.shownTile[c] += 1
                    c = c[0] + chr(ord(c[1]) + 1)
            else:
                self.playerData[player].tile.append(self.lastTile)
                c = outputList[1]
                if c[0] not in 'WBT' or c[0] != self.lastTile[0] or abs(ord(c[1]) - ord(self.lastTile[1])) > 1:
                    self.playerError(player, "吃非数字牌或数字不匹配")
                c = c[0] + chr(ord(c[1]) - 1)
                for i in [-1, 0, 1]:
                    self.shownTile[c] += 1
                    if c not in self.playerData[player].tile:
                        self.playerError(player, "无搭子吃牌")
                    self.playerData[player].tile.remove(c)
                    c = c[0] + chr(ord(c[1]) + 1)
            self.lastOp = "CHI"
            self.tileCHI = outputList[1]
            self.playerData[player].pack.append(Pack("CHI", self.tileCHI, ord(self.lastTile[1]) - ord(outputList[1][1]) + 1))
            self.lastTile = outputList[2]
            if player != self.id:
                self.playerData[player].tile.pop(-1)
            else:
                if self.lastTile not in self.playerData[player].tile:
                    self.playerError(player, "打出非手牌 " + self.lastTile)
                self.playerData[player].tile.remove(self.lastTile)
            self.roundStage = 4 + player
            return True
        return False

    def checkInputGANG(self, response: str, player: int):
        if self.lastBUGANG and self.roundStage % 4 != player and response == "HU":
            self.checkHu(player, self.canHu[player])
        if response == "PASS":
            return
        self.playerError(player, "非法操作 " + response)

    def infer_response(self, request: str):
        inferred_response = ['PASS'] * 4
        request = request.split(' ')
        if request[0] == '3' and request[2] in ['PLAY', 'CHI', 'PENG', 'GANG', 'BUGANG']:
            player = int(request[1])
            inferred_response[player] = ' '.join(request[2:])
            if request[2] == 'GANG' and self.lastOp == 'DRAW':
                inferred_response[player] += ' ??'
        return inferred_response


    def roundOutput(self, request: str):
        request = request.split(' ')
        if self.roundStage == -1:
            hua = [int(i) for i in request[1:5]]
            for i in range(4):
                while len(self.playerData[i].tile) < 13 + hua[i]:
                    nextTile = self.playerData[i].pTileWall.pop(-1)
                    self.playerData[i].tile.append(nextTile)
            self.playerData[self.id].tile = request[5:18]
            self.playerData[0].flower = request[18:18+hua[0]]
            self.playerData[1].flower = request[18+hua[0]:18+hua[1]]
            self.playerData[2].flower = request[18+hua[1]:18+hua[2]]
            self.playerData[3].flower = request[18+hua[2]:18+hua[3]]
        elif self.roundStage >= 0 and self.roundStage < 4:
            tw = self.playerData[self.roundStage % 4].pTileWall
            if tw == []:
                pass
            self.lastTile = tw.pop(-1)
            if request[:3] == ['3', str(self.roundStage % 4), 'BUHUA']:
                self.lastTile = request[3]
            elif self.roundStage % 4 == self.id:
                assert request[0] == '2'
                self.lastTile = request[1]
            else:
                assert request[:3] == ['3', str(self.roundStage % 4), 'DRAW']
            if self.lastTile[0] == 'H':
                self.lastOp = "BUHUA"
                self.playerData[self.roundStage % 4].flower.append(self.lastTile)
            else:
                self.lastOp = "DRAW"
                if self.roundStage % 4 == self.id:
                    self.canHu = self.checkHu(self.id, -1)
        elif self.roundStage >= 4 and self.roundStage < 8:
            if self.playerData[(self.lastRoundStage + 1) % 4].pTileWall == [] and self.lastOp in ["CHI", "PENG"]:
                self.playerError(self.roundStage % 4, "终局吃碰 " + self.lastOp)
            if self.roundStage % 4 != self.id:
                self.canHu = self.checkHu(self.id, -1)
            assert request[:2] == ['3', str(self.roundStage % 4)] and request[2] in ['CHI', 'PENG', 'PLAY']
        else:
            if self.playerData[(self.lastRoundStage + 1) % 4].pTileWall == [] and self.lastOp in ["GANG", "BUGANG"]:
                self.playerError(self.roundStage % 4, "终局杠牌 " + self.lastOp)
            if self.lastOp != "GANG" and self.lastBUGANG:
                if self.roundStage % 4 != self.id:
                    self.canHu = self.checkHu(self.id, -1)
            assert request[:2] == ['3', str(self.roundStage % 4)] and request[2] in ["GANG", "BUGANG"]
        shanten = [MahjongShanten(
            hand = tuple(self.playerData[i].tile),
            pack = tuple(v.as_tuple() for v in self.playerData[i].pack))
            if i == self.id else 0
            for i in range(4)
        ]
        self.vec_data.sync(shanten, [self.id])
        if self.canHu:
            self.vec_data.enable_hu(self.id)
        self.vec_data.enable_pass()
        if self.lastOp == "DRAW" and self.roundStage % 4 == self.id:
            self.vec_data.check_able_play(self.roundStage % 4, self.lastTile)
        if self.roundStage >= 4 and self.roundStage < 8:
            self.vec_data.check_able_ming(self.roundStage % 4, self.lastTile)
        

    def roundInput(self, response: list):
        self.lastRoundStage = self.roundStage
        if self.roundStage < 0:
            for i in range(4):
                self.checkInputPASS(response[i], i)
            self.roundStage += 1
        elif self.roundStage >= 0 and self.roundStage < 4:
            for i in range(4):
                if self.lastOp == "BUHUA" or self.roundStage != i:
                    self.checkInputPASS(response[i], i)
                else:
                    self.checkInputDRAW(response[i], i)
            self.lastBUGANG = self.currBUGANG
            self.lastGANG = self.currGANG
            self.lastANGANG = self.currANGANG
            self.currBUGANG = False
            self.currGANG = False
            self.currANGANG = False
        elif self.roundStage >= 4 and self.roundStage < 8:
            self.checkInputPASS(response[self.roundStage % 4], self.roundStage % 4)
            for i in range(1, 4):
                self.checkInputPLAY1(response[(self.roundStage + i) % 4], (self.roundStage + i) % 4)
            b_pass = True
            for i in range(4):
                if b_pass and self.roundStage != i + 4:
                    b_pass = not self.checkInputPLAY2(response[i], i)
            for i in range(4):
                if b_pass and self.roundStage != i + 4:
                    b_pass = not self.checkInputPLAY3(response[i], i)
            if b_pass:
                self.roundStage = (self.roundStage + 1) % 4
                self.shownTile[self.lastTile] += 1
        else:
            for i in range(4):
                self.checkInputGANG(response[i], (self.roundStage + i) % 4)
            self.roundStage -= 8


def main():
    device = torch.device('cpu')
    net = Net(Bot.state_shape, Bot.extra_shape, Bot.action_shape, device)
    actor = Actor(net, Bot.state_shape, Bot.extra_shape, Bot.action_shape, device)
    state_dict = {
        k[6:]: v
        for k, v in torch.load('./data/policy.pth', map_location='cpu').items() if k[:6] == 'actor.'
    }
    actor.load_state_dict(state_dict)
    actor.eval()
    full_input = json.loads(input())
    all_requests = full_input["requests"]
    all_responses = full_input["responses"]
    if len(all_requests) == 1:
        action = 'PASS'
    else:
        bot = Bot()
        obs = bot.get_obs(all_requests, all_responses)
        dist = torch.distributions.Categorical
        with torch.no_grad():
            logits, _ = actor(obs, None)
        action = torch.argmax(dist(logits).probs, dim=-1).item()
        action = bot.vec_data.realize(action)
    print(json.dumps({
        "response": action,
        "debug": "",
        "data": "",
        "globaldata": ""
    }))

if __name__ == "__main__":
    main()
