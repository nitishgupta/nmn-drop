import os
import json


class WDWDoc():
    def __init__(self, **kwargs):
        self.qid = kwargs['qid']
        self.qleftContext = kwargs['qleftContext']
        self.qrightContext = kwargs['qrightContext']
        self.contextId = kwargs['contextId']
        self.contextPara = kwargs['contextPara']
        self.correctChoice = kwargs['correctChoice']
        self.candidateChoices = kwargs['candidateChoices']

    def tojson(self) -> str:
        jsondict = self.__dict__
        jsonstr = json.dumps(jsondict)
        return jsonstr
