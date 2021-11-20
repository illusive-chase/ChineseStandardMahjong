# -*- coding: utf-8 -*-

__all__ = ('Runner',)

from MahjongGB import MahjongFanCalculator, MahjongShanten
from utils.vec_data import VecData
from utils.policy import random_policy, imitation_policy
import numpy as np
import gym
from copy import deepcopy


def visualize(tile):
    mode = '\033[1;{};40m'.format({
        'W': 37,
        'T': 32,
        'B': 31,
        'F': 36,
        'J': 35,
        '?': 34
    }[tile[0]])
    default_mode = '\033[0m'
    return mode + tile[1:] + default_mode

def visualize_pack(pack):
    mode = '\033[1;{};40m'.format({
        'W': 37,
        'T': 32,
        'B': 31,
        'F': 36,
        'J': 35,
        '?': 34
    }[pack.tile[0]])
    default_mode = '\033[0m'
    if pack.type == "CHI":
        num = int(pack.tile[1:])
        tiles = ''.join([str(num - 1), str(num), str(num + 1)])
    elif pack.type == "GANG":
        tiles = 4 * pack.tile[1:]
        if pack.isANGANG:
            tiles = '[' + tiles + ']'
    elif pack.type == "PENG":
        tiles = 3 * pack.tile[1:]
    else:
        raise ValueError    
    return mode + tiles + default_mode


class FinishError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Runner(gym.Env):

    state_shape = VecData.state_shape
    action_shape = VecData.action_shape

    def __init__(self, other_policy, verbose, seed=1, eval=False):
        self.verbose = verbose
        self.other_policy = other_policy
        self.action_space = gym.spaces.Discrete(self.action_shape[1])
        self.eval = eval
        self.tileWallDummy = None
        self.seed(seed)
        # self.no_need_other_obs = type(other_policy) is imitation_policy
        self.no_need_other_obs = False


    def reset(self, init_data=None):
        self.display = []
        self.canHu = [-4] * 4
        self.rew = np.zeros((4,), dtype=np.float64)

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

        self.wait_to_play = None

        if init_data is None:
            not_reset = self.tileWallDummy is not None
            self.quan = self.quan if not_reset else np.random.randint(0, 3)
            self.id = (self.id + 1) % 4 if not_reset else np.random.randint(0, 3)
            self.other = [i for i in range(4) if i != self.id]
            self.tileWall = []
            for k in "WBT":
                for i in range(1, 10):
                    for j in range(4):
                        self.tileWall.append(k + str(i))
            for i in range(1, 5):
                for j in range(4):
                    self.tileWall.append("F" + str(i))
            for i in range(1, 4):
                for j in range(4):
                    self.tileWall.append("J" + str(i))
            
            np.random.shuffle(self.tileWall)
            if self.eval:
                if not_reset:
                    self.tileWall = deepcopy(self.tileWallDummy)
                else:
                    self.tileWallDummy = deepcopy(self.tileWall)
        else:
            self.quan = init_data.quan
            self.id = init_data.id
            self.other = [i for i in range(4) if i != self.id]
            self.tileWall = init_data.tile_wall

        self.vec_data = VecData(self.quan, (self.other if self.no_need_other_obs else []))
        self.playerData, self.shownTile, self.str2tile, self.tile2str = self.vec_data.connect()
        if init_data is not None:
            self.tileWall = [self.tile2str[tile] if tile < 34 else '??' for tile in self.tileWall]
        partLength = len(self.tileWall) // 4
        for i in range(4):
            for j in range(partLength):
                self.playerData[i].pTileWall.append(self.tileWall[partLength * i + j])

        if self.verbose:
            self.display.append("初始化：SEED {}".format(self.randSeed))
            self.display.append("\t场风：{}".format("东南西北"[self.quan]))
            self.display.append("\t牌山：" + ''.join(map(visualize, self.tileWall)))

        

        for _ in range(2):
            self.roundInput(["PASS"] * 4)
            self.canHu = [0] * 4
            self.roundOutput()
        train, self.fixData = self.vec_data.get_obs(self.id, self.other)
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
        raise ValueError

    def checkHu(self, player: int, re: int):
        if re == -1:
            shanten = MahjongShanten(
                hand = tuple(self.playerData[player].tile),
                pack = tuple(v.as_tuple(player) for v in self.playerData[player].pack)
            )
            if shanten > 0:
                return 0

            try:
                fan_table = MahjongFanCalculator(
                    hand = tuple(self.playerData[player].tile),
                    pack = tuple(v.as_tuple(player) for v in self.playerData[player].pack),
                    winTile = self.lastTile,
                    flowerCount = len(self.playerData[player].flower),
                    isSelfDrawn = self.roundStage == player,
                    is4thTile = self.shownTile[self.str2tile[self.lastTile]] == 3,
                    isAboutKong = self.lastBUGANG or self.lastANGANG or self.currGANG,
                    isWallLast = self.playerData[(self.roundStage + 1) % 4].pTileWall == [],
                    seatWind = player,
                    prevalentWind  = self.quan,
                    verbose = False
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
                hand = tuple(self.playerData[player].tile)
                hand = hand if self.roundStage != player else hand[:-1]
                fan_table = MahjongFanCalculator(
                    hand = hand,
                    pack = tuple(v.as_tuple(player) for v in self.playerData[player].pack),
                    winTile = self.lastTile,
                    flowerCount = len(self.playerData[player].flower),
                    isSelfDrawn = self.roundStage == player,
                    is4thTile = self.shownTile[self.str2tile[self.lastTile]] == 3,
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
                    self.rew[i] = 3 * 8 + re
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
            self.lastTile = outputList[1]
            if outputList[0] == "PLAY":
                if self.playerData[player].play(self.lastTile):
                    self.lastOp = "PLAY"
                    self.roundStage += 4
                    return
            elif outputList[0] == "GANG":
                if self.playerData[player].pTileWall == [] or self.playerData[(player + 1) % 4].pTileWall == []:
                    self.playerError(player, "终局杠牌")
                if not self.playerData[player].angang(self.lastTile, player):
                    self.playerError(player, "无牌暗杠")
                self.lastOp = "GANG"
                self.currANGANG = True
                self.currGANG = False
                self.lastGANG = False
                self.currBUGANG = False
                self.lastBUGANG = False
                self.roundStage = player + 8
                return
            elif outputList[0] == "BUGANG":
                if self.lastTile not in self.playerData[player].tile:
                    self.playerError(player, "无牌补杠")
                if self.playerData[player].bugang(self.lastTile):
                    self.vec_data.show(self.lastTile, 1)
                    self.lastOp = "BUGANG"
                    self.currBUGANG = True
                    self.currANGANG = False
                    self.currGANG = False
                    self.lastGANG = False
                    self.lastBUGANG = False
                    self.roundStage = player + 8
                    return
                else:
                    self.playerError(player, "无补杠对应刻子")
        self.playerError(player, "非法操作 " + response)
                
            

    def checkInputPLAY1(self, response: str, player: int):
        if response == "HU":
            self.checkHu(player, self.canHu[player])

    def checkInputPLAY2(self, response: str,  player: int):
        outputList = response.split(' ')
        if response == "PASS":
            return False
        elif response == "GANG":
            if not self.playerData[player].gang(self.lastTile, self.roundStage % 4):
                self.playerError(player, "无暗刻杠牌")
            self.vec_data.show(self.lastTile, 4)
            self.lastOp = "GANG"
            self.currGANG = True
            self.currBUGANG = False
            self.currANGANG = False
            self.lastGANG = False
            self.lastBUGANG = False
            self.roundStage = player + 8
            return True
        elif len(outputList) == 2:
            if outputList[0] == "PENG":
                if not self.playerData[player].peng(self.lastTile, self.roundStage % 4):
                    self.playerError(player, "无对子碰牌")
                self.vec_data.show(self.lastTile, 3)
                self.lastOp = "PENG"
                self.lastTile = outputList[1]
                if self.lastTile == '??':
                    self.wait_to_play = player
                else:
                    if not self.playerData[player].play(self.lastTile):
                        self.playerError(player, "打出非手牌 " + self.lastTile)
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
            self.playerData[player].draw(self.lastTile)
            c = outputList[1]
            if c[0] not in 'WBT' or c[0] != self.lastTile[0] or abs(ord(c[1]) - ord(self.lastTile[1])) > 1:
                self.playerError(player, "吃非数字牌或数字不匹配")
            if not self.playerData[player].chi(c, self.lastTile):
                self.playerError(player, "无搭子吃牌")
            tile_t = self.str2tile[c]
            self.vec_data.show_range(tile_t - 1, tile_t + 2, 1)
            self.lastOp = "CHI"
            self.tileCHI = outputList[1]
            self.lastTile = outputList[2]
            if self.lastTile == '??':
                self.wait_to_play = player
            else:
                if not self.playerData[player].play(self.lastTile):
                    self.playerError(player, "打出非手牌 " + self.lastTile)
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
        if self.wait_to_play is not None:
            self.vec_data.sync(idxs=[self.wait_to_play])
            self.vec_data.check_able_play(self.wait_to_play, False)
            return
        post_fn = lambda :None
        if self.roundStage == -1:
            if self.verbose:
                self.display.append("发牌：")
            for i in range(4):
                while len(self.playerData[i].tile) < 13:
                    nextTile = self.playerData[i].pTileWall.pop(-1)
                    if nextTile[0] == 'H':
                        self.playerData[i].flower.append(nextTile)
                    else:
                        self.playerData[i].draw(nextTile)
            if self.verbose:
                for i in range(4):
                    self.display.append("\tPLAYER {} 花牌 {} 手牌 {}".format(i, self.playerData[i].flower, ''.join(map(visualize, sorted(self.playerData[i].tile)))))
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
                post_fn = lambda : self.playerData[self.roundStage].draw(self.lastTile)
        elif self.roundStage >= 4 and self.roundStage < 8:
            if self.playerData[(self.lastRoundStage + 1) % 4].pTileWall == [] and self.lastOp in ["CHI", "PENG"]:
                self.playerError(self.roundStage % 4, "终局吃碰 " + self.lastOp)
            if self.verbose and self.wait_to_play is None:
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
        post_fn()
        self.vec_data.sync()
        for i in range(4):
            if self.canHu[i] > 0:
                self.vec_data.enable_hu(i)
        self.vec_data.enable_pass()
        if self.lastOp == "DRAW":
            self.vec_data.check_able_play(self.roundStage % 4)
        if self.roundStage >= 4 and self.roundStage < 8:
            self.vec_data.check_able_ming(self.roundStage % 4, self.lastTile)
        

    def roundInput(self, response: list):
        if self.wait_to_play is not None:
            player = self.wait_to_play
            outputList = response[player].split()
            self.lastTile = outputList[1]
            assert outputList[0] == "PLAY"
            if not self.playerData[player].play(self.lastTile):
                self.playerError(player, "打出非手牌 " + self.lastTile)
            # self.vec_data.show(self.lastTile, 1)
            self.wait_to_play = None
            return

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
                if self.wait_to_play is None:
                    self.vec_data.show(self.lastTile, 1)
        else:
            for i in range(4):
                self.checkInputGANG(response[(self.roundStage + i) % 4], (self.roundStage + i) % 4)
            self.roundStage -= 8


    def step(self, raw_action):
        fix_action = self.other_policy(self.fixData)
        action = [raw_action if i == self.id else fix_action[self.other.index(i)] for i in range(4)]
        real_action = [self.vec_data.realize(action[i]) for i in range(4)]
        while True:
            try:
                self.roundInput(real_action)
                self.canHu = [-4] * 4
                self.roundOutput()
            except FinishError:
                return self.vec_data.get_obs(self.id, other=[])[0], self.rew[self.id], np.array(True), {}
            finally:
                pass
            if self.wait_to_play is None:
                train, self.fixData = self.vec_data.get_obs(self.id, self.other)
            else:
                train, _ = self.vec_data.get_obs(self.wait_to_play, other=[])
                if self.wait_to_play != self.id:
                    real_action[self.wait_to_play] = self.vec_data.realize(self.other_policy(train))
                    continue
            return train, 0.0, np.asarray(False), {}
        
    def render(self):
        if self.verbose:
            for i in range(4):
                ptile = list(sorted(self.playerData[i].tile))
                if self.lastOp == "DRAW" and i == self.roundStage:
                    serve = visualize(self.lastTile)
                    ptile.pop(ptile.index(self.lastTile))
                else:
                    serve = ''
                self.display.append("PLAYER {} REST {}\n\tHAND {} {}\n\tPACK {}".format(
                    i,
                    len(self.playerData[i].pTileWall),
                    ''.join(map(visualize, ptile)),
                    serve,
                    ' '.join(map(visualize_pack, self.playerData[i].pack))
                ))
            print("\n".join(self.display))
            self.display = []

    def close(self):
        pass

    def seed(self, randSeed):
        self.randSeed = randSeed
        np.random.seed(randSeed)
