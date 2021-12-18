# -*- coding: utf-8 -*-

__all__ = ('Bot',)


from utils.vec_data import VecData
from MahjongGB import MahjongFanCalculator, MahjongShanten
import json


class Bot:

    state_shape = VecData.state_shape
    action_shape = VecData.action_shape

    def __init__(self):
        pass
        
    def get_obs(self, requests, responses):

        _, id, quan = requests[0].split(' ')
        requests = requests[1:]

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

        self.quan = int(quan)
        self.id = int(id)
        self.other = [i for i in range(4) if i != self.id]

        self.wait_to_play = None

        self.vec_data = VecData(self.quan, self.other)
        self.playerData, self.shownTile, self.str2tile, self.tile2str = self.vec_data.connect()
        self.canHu = 0
        

        partLength = 136 // 4
        for i in range(4):
            for j in range(partLength):
                self.playerData[i].add_to_wall('??')

        for response, request in zip(responses, requests):
            inferred_response = self.infer_response(request)
            inferred_response[self.id] = response
            self.roundInput(inferred_response)
            self.roundOutput(request)


        # out->request->response->input->out->request
        #                  ^              ^      |
        #                  |              |      |
        #                  -----------------------

        return self.vec_data.get_obs(self.id, other=[])[0]


    def playerError(self, player: int, code: str):
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
                hand = tuple(self.playerData[player].tile)
                hand = hand if self.roundStage != player else hand[:-1]
                fan_table = MahjongFanCalculator(
                    hand = hand,
                    pack = tuple(p.as_tuple(player) for p in self.playerData[player].pack),
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
                self.checkHu(player, self.canHu)
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
                if self.playerData[player].angang(self.lastTile, player):
                    self.lastOp = "GANG"
                    self.currANGANG = True
                    self.currGANG = False
                    self.lastGANG = False
                    self.currBUGANG = False
                    self.lastBUGANG = False
                    self.roundStage = player + 8
                    return
            elif outputList[0] == "BUGANG":
                if self.playerData[player].bugang(self.lastTile):
                    for i, pack in enumerate(self.playerData[player].pack):
                        self.vec_data.show(self.lastTile, 1)
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
            assert player == self.id
            self.checkHu(player, self.canHu)

    def checkInputPLAY2(self, response: str, player: int):
        outputList = response.split(' ')
        if response == "PASS":
            return False
        elif response == "GANG":
            if self.playerData[player].gang(self.lastTile, self.roundStage % 4):
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
                if self.playerData[player].peng(self.lastTile, self.roundStage % 4):
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
            if self.playerData[player].chi(c, self.lastTile):
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
            assert player == self.id
            self.checkHu(player, self.canHu)
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
        if self.wait_to_play is not None:
            self.vec_data.sync(idxs=[self.wait_to_play])
            self.vec_data.check_able_play(self.wait_to_play, False)
            return
        
        post_fn = lambda :None
        request = request.split(' ')
        if self.roundStage == -1:
            hua = [int(i) for i in request[1:5]]
            for i in range(13):
                for j in self.other:
                    nextTile = self.playerData[j].get_from_wall()
                    self.playerData[j].draw(nextTile)
                self.playerData[self.id].get_from_wall()
                self.playerData[self.id].draw(request[5 + i])
            self.playerData[0].flower = request[18:18+hua[0]]
            self.playerData[1].flower = request[18+hua[0]:18+hua[1]]
            self.playerData[2].flower = request[18+hua[1]:18+hua[2]]
            self.playerData[3].flower = request[18+hua[2]:18+hua[3]]
        elif self.roundStage >= 0 and self.roundStage < 4:
            tw = self.playerData[self.roundStage % 4].pTileWall
            if tw == []:
                pass
            self.lastTile = self.playerData[self.roundStage % 4].get_from_wall()
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
                post_fn = lambda : self.playerData[self.roundStage].draw(self.lastTile)
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
        post_fn()
        self.vec_data.sync()
        if self.canHu:
            self.vec_data.enable_hu(self.id)
        self.vec_data.enable_pass()
        if self.lastOp == "DRAW" and self.roundStage % 4 == self.id:
            self.vec_data.check_able_play(self.roundStage % 4)
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
                if self.wait_to_play is None:
                    self.vec_data.show(self.lastTile, 1)
        else:
            for i in range(4):
                self.checkInputGANG(response[(self.roundStage + i) % 4], (self.roundStage + i) % 4)
            self.roundStage -= 8


    def stepOneRound(self, policy):
        full_input = json.loads(input())
        all_requests = full_input["requests"]
        all_responses = full_input["responses"]
        if len(all_requests) == 1:
            action = 'PASS'
        else:
            obs = self.get_obs(all_requests, all_responses)
            action = self.vec_data.realize(policy(obs))
            if '??' in action:
                self.roundInput([action if i == self.id else 'PASS' for i in range(4)])
                self.roundOutput(None)
                obs = self.vec_data.get_obs(self.id, other=[])[0]
                extra_action = self.vec_data.realize(policy(obs))
                assert extra_action[:5] == 'PLAY '
                action = action.replace('??', extra_action[5:])
        print(json.dumps({
            "response": action,
            "debug": "",
            "data": "",
            "globaldata": ""
        }))

