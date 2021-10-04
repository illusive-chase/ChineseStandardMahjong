# -*- coding: utf-8 -*-
from MahjongGB import MahjongFanCalculator, MahjongShanten
import random
from test import VecData
from test import Pack, PlayerData
import numpy as np
import gym
from tianshou.data import Batch
from copy import deepcopy

def visualize(tile):
    mode = '\033[1;{};40m'.format({
        'W': 37,
        'T': 32,
        'B': 31,
        'F': 36,
        'J': 35
    }[tile[0]])
    default_mode = '\033[0m'
    return mode + tile[1:] + default_mode

def visualize_pack(pack):
    mode = '\033[1;{};40m'.format({
        'W': 37,
        'T': 32,
        'B': 31,
        'F': 36,
        'J': 35
    }[pack.tile[0]])
    default_mode = '\033[0m'
    if pack.type == "CHI":
        num = int(pack.tile[1:])
        tiles = ''.join([str(num - 1), str(num), str(num + 1)])
    elif pack.type == "GANG":
        tiles = 4 * pack.tile[1:]
    elif pack.type == "PENG":
        tiles = 3 * pack.tile[1:]
    else:
        raise ValueError    
    return mode + tiles + default_mode


class FinishError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Referee(gym.Env):

    state_shape = VecData.state_shape
    extra_shape = VecData.extra_shape
    action_shape = VecData.action_shape

    def __init__(self, fixPolicy, verbose, eval=False):
        self.verbose = verbose
        self.randSeed = 1
        self.fixPolicy = fixPolicy
        self.action_space = gym.spaces.Discrete(self.action_shape[1])
        self.eval = eval
        self.tileWallDummy = None

    def reset(self, inputValue: dict = {}):
        self.display = []
        self.canHu = [-4] * 4
        self.rew = np.zeros((4,), dtype=np.float64)

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

        not_reset = self.tileWallDummy is not None

        # quan: int
        self.quan = self.quan if not_reset else random.randint(0, 3)
        # tileWall: list[str]
        self.tileWall = []
        # self.shownTile: dict[str:int]
        self.shownTile = {}
        self.str2tile = {}

        for k in "WBT":
            for i in range(1, 10):
                for j in range(4):
                    self.tileWall.append(k + str(i))
                self.shownTile[k + str(i)] = 0
        for i in range(1, 5):
            for j in range(4):
                self.tileWall.append("F" + str(i))
            self.shownTile["F" + str(i)] = 0
        for i in range(1, 4):
            for j in range(4):
                self.tileWall.append("J" + str(i))
            self.shownTile["J" + str(i)] = 0

        self.tile2str = list(self.shownTile.keys())
        for idx, tile in enumerate(self.tile2str):
            self.str2tile[tile] = idx

        self.vec_data = VecData(self.shownTile, self.playerData, self.str2tile, self.tile2str, self.quan)
        
        
        random.shuffle(self.tileWall)
        if self.eval:
            if not_reset:
                self.tileWall = deepcopy(self.tileWallDummy)
            else:
                self.tileWallDummy = deepcopy(self.tileWall)
        if self.verbose:
            self.display.append("初始化：SEED {}".format(self.randSeed))
            self.display.append("\t场风：{}".format("东南西北"[self.quan]))
            self.display.append("\t牌山：" + ''.join(map(visualize, self.tileWall)))

        partLength = len(self.tileWall) // 4
        for i in range(4):
            for j in range(partLength):
                self.playerData[i].pTileWall.append(self.tileWall[partLength * i + j])

        for _ in range(2):
            self.roundInput(["PASS"] * 4)
            self.canHu = [0] * 4
            self.roundOutput()
        train, self.fixData = self.vec_data.get_obs()
        return train


    def playerError(self, player: int, code: str):
        if self.verbose:
            self.display.append("结束：PLAYER {} ERROR {}".format(player, code))
            for i in range(4):
                if (i == player):
                    self.display.append("\tPLAYER {} SCORE -30".format(i))
                else:
                    self.display.append("\tPLAYER {} SCORE +10".format(i))
        for i in range(4):
            if (i == player):
                self.rew[i] = -30.
            else:
                self.rew[i] = 10.
        raise FinishError

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
                if self.lastTile in self.playerData[player].tile:
                    self.playerData[player].tile.remove(self.lastTile)
                    self.lastOp = "PLAY"
                    self.roundStage += 4
                    return
            elif outputList[0] == "GANG":
                if self.playerData[player].pTileWall == [] or self.playerData[(player + 1) % 4].pTileWall == []:
                    self.playerError(player, "终局杠牌")
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

    def checkInputPLAY2(self, response: str,  player: int):
        outputList = response.split(' ')
        if response == "PASS":
            return False
        elif response == "GANG":
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
                for i in range(2):
                    if self.lastTile not in self.playerData[player].tile:
                        self.playerError(player, "无对子碰牌")
                    self.playerData[player].tile.remove(self.lastTile)
                self.shownTile[self.lastTile] += 3
                self.lastOp = "PENG"
                self.playerData[player].pack.append(Pack("PENG", self.lastTile, self.roundStage % 4))
                self.lastTile = outputList[1]
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

    def roundOutput(self):
        if self.roundStage == -1:
            if self.verbose:
                self.display.append("发牌：")
            for i in range(4):
                while len(self.playerData[i].tile) < 13:
                    nextTile = self.playerData[i].pTileWall.pop(-1)
                    if nextTile[0] == 'H':
                        self.playerData[i].flower.append(nextTile)
                    else:
                        self.playerData[i].tile.append(nextTile)
            if self.verbose:
                for i in range(4):
                    self.playerData[i].tile.sort()
                    self.display.append("\tPLAYER {} 花牌 {} 手牌 {}".format(i, self.playerData[i].flower, ''.join(map(visualize, self.playerData[i].tile))))
        elif self.roundStage >= 0 and self.roundStage < 4:
            tw = self.playerData[self.roundStage % 4].pTileWall
            if tw == []:
                if self.verbose:
                    self.display.append("结束：荒牌")
                raise FinishError
            self.lastTile = tw.pop(-1)
            if self.lastTile[0] == 'H':
                self.lastOp = "BUHUA"
                if self.verbose:
                    self.display.append("PLAYER {} 补花 {}".format(self.roundStage, self.lastTile))
                self.playerData[self.roundStage % 4].flower.append(self.lastTile)
            else:
                self.lastOp = "DRAW"
                if self.verbose:
                    self.display.append("PLAYER {} 摸牌 {}".format(self.roundStage, visualize(self.lastTile)))
                self.canHu[self.roundStage] = self.checkHu(self.roundStage, -1)
        elif self.roundStage >= 4 and self.roundStage < 8:
            if self.playerData[(self.lastRoundStage + 1) % 4].pTileWall == [] and self.lastOp in ["CHI", "PENG"]:
                self.playerError(self.roundStage % 4, "终局吃碰 " + self.lastOp)
            if self.verbose:
                if self.lastOp == "CHI":
                    self.display.append("PLAYER {} 吃牌 {} 出牌 {}".format(self.roundStage % 4, visualize(self.tileCHI), visualize(self.lastTile)))
                else:
                    self.display.append("PLAYER {} {} {}".format(self.roundStage % 4, {"PENG": "碰牌 出牌", "PLAY": "出牌"}[self.lastOp], visualize(self.lastTile)))
            for i in range(4):
                if self.roundStage % 4 != i:
                    self.canHu[i] = self.checkHu(i, -1)
        else:
            if self.playerData[(self.lastRoundStage + 1) % 4].pTileWall == [] and self.lastOp in ["GANG", "BUGANG"]:
                self.playerError(self.roundStage % 4, "终局杠牌 " + self.lastOp)
            if self.lastOp != "GANG" and self.lastBUGANG:
                for i in range(4):
                    if self.roundStage % 4 != i:
                        self.canHu[i] = self.checkHu(i, -1)
            if self.verbose:
                self.display.append("PLAYER {} {}".format(self.roundStage % 4, ("杠牌" if self.lastOp == "GANG" else "补杠")))
        self.shanten = [MahjongShanten(
            hand = tuple(self.playerData[i].tile),
            pack = tuple(v.as_tuple() for v in self.playerData[i].pack))
            for i in range(4)
        ]
        self.vec_data.sync(self.shanten)
        for i in range(4):
            if self.canHu[i] > 0:
                self.vec_data.enable_hu(i)
        self.vec_data.enable_pass()
        if self.lastOp == "DRAW":
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


    def step(self, action):
        fix_action = self.fixPolicy(Batch({'obs': self.fixData})).act
        action = [action if i == 0 else fix_action[i - 1] for i in range(4)]
        real_action = [self.vec_data.realize(action[i]) for i in range(4)]
        try:
            self.roundInput(real_action)
            self.canHu = [-4] * 4
            self.roundOutput()
        except FinishError:
            rew = self.rew[0]
            rew = -2 if rew < 0 else (-1 if rew == 0 else (0.5 * self.rew[0]) ** 0.5)
            return self.vec_data.get_obs()[0], rew, np.array(True), {}
        finally:
            pass
        train, self.fixData = self.vec_data.get_obs()
        st = self.shanten[0]
        rew = 1 / (2 + st) * 0.01
        return train, (rew if self.eval else 0.0), np.array(False), {}

        
    def render(self):
        if self.verbose:
            for i in range(4):
                self.playerData[i].tile.sort()
                self.display.append("PLAYER {} REST {}\n\tHAND {} {}\n\tPACK {}".format(
                    i,
                    len(self.playerData[i].pTileWall),
                    ''.join(map(visualize, self.playerData[i].tile)),
                    visualize(self.lastTile) if self.lastOp == "DRAW" and i == self.roundStage else '',
                    ' '.join(map(visualize_pack, self.playerData[i].pack))
                ))
            print("\n".join(self.display))
            self.display = []

    def close(self):
        pass

    def seed(self, randSeed):
        self.randSeed = randSeed
        random.seed(randSeed)
        np.random.seed(randSeed)
