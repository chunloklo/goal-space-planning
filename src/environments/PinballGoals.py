import random
import argparse, os
from re import S
import numpy as np
from RlGlue import BaseEnvironment
from itertools import *

class PinballGoals():

    termination_radius = 0.04
    initiation_radius = 0.48
    speed_radius = 0.2
    
    # Caching calculation for functions
    termination_radius_squared = np.square(termination_radius)
    initiation_radius_squared = np.square(initiation_radius)
    speed_radius_squared = np.square(speed_radius)

    def __init__(self):
        self.goals = []
        # n is the number of goals "per side". Total number of goals is n x n
        n = 4
        border = 0.06

        for y in np.linspace(0 + border, 1 - border, n):
            for x in np.linspace(0 + border, 1 - border, n):
                self.goals.append([x,y])

        self.goals = np.array(self.goals)
        self.num_goals = self.goals.shape[0]

        # Shifting away some goals so they aren't on top of obstacles
        self.goals[9] += [-0.06, 0.0]
        self.goals[10] += [-0.08, 0.0]
        self.goals[11] += [0.0, 0.06]

        self.goal_speeds = np.zeros(self.goals.shape)
        
    def goal_termination(self, s, a, sp):
        # [chunlok 2022-04-15] It's important here that sp is used instead of s because the way the environment works.
        # This is so that the reward is obtained by the agent right AFTER the option terminates, rather than on the transition on which the option terminates
        state_close = np.sum(np.power(self.goals - sp[:2], 2), axis=1) <= self.termination_radius_squared
        speed_close = np.sum(np.power(sp[2:] - self.goal_speeds, 2), axis=1) <= self.speed_radius_squared
        terms = np.logical_and(state_close, speed_close)
        return terms

    def goal_initiation(self, s):
        state_close = np.sum(np.power(self.goals - s[:2], 2), axis=1) <= self.initiation_radius_squared

        # Checks for termination. Don't let a goal wrap around to itself
        goal_term = self.goal_termination(None, None, s)
        
        return np.logical_and(state_close, ~goal_term)

class PinballOracleGoals(PinballGoals):
    def __init__(self):
        super().__init__()

        self.termination_radius = 0.04
        self.initiation_radius = 0.4
        self.speed_radius = 2.0
        
        # Caching calculation for functions
        self.termination_radius_squared = np.square(self.termination_radius)
        self.initiation_radius_squared = np.square(self.initiation_radius)
        self.speed_radius_squared = np.square(self.speed_radius)

        # Keeping the terminal goal
        self.goals = [self.goals[3]]
        self.goal_speeds = [[0.0, 0.0]]

        trajectory = [
            [0.2, 0.9, 0.0, 0.0],
            [0.2, 0.8960000000000005, 0.0, -0.199],
            [0.2, 0.8880199999999997, 0.0, -0.397005],
            [0.2, 0.8760798999999989, 0.0, -0.594019975],
            [0.20400000000000013, 0.8641995004999995, 0.199, -0.5910498751250001],
            [0.20798000000000022, 0.8483785029974992, 0.19800500000000001, -0.7870946257493752],
            [0.21594010000000036, 0.8326366104825126, 0.39601497500000005, -0.7831591526206283],
            [0.22786039950000042, 0.8169734274301003, 0.5930349001250002, -0.7792433568575251],
            [0.24372109750250043, 0.8013885602929497, 0.789069725624375, -0.7753471400732375],
            [0.26350249201498754, 0.7858816174914854, 0.9841243769962531, -0.7714704043728713],
            [0.2831849795549124, 0.7664522094040286, 0.9792037551112718, -0.9666130523510069],
            [0.3031849795549124, 0.7471199483570082, 0.995, -0.9617799870892518],
            [0.3230849795549129, 0.731884348615223, 0.990025, -0.7579710871538056],
            [0.342885479554913, 0.7207249268721464, 0.985074875, -0.5551812317180365],
            [0.36258697705491283, 0.7256974641161994, 0.980149500625, 0.35340532555944626],
            [0.38258697705491285, 0.7327655706273893, 0.995, 0.35163829893164905],
            [0.40258697705491286, 0.7397983366060233, 0.995, 0.34988010743699083],
            [0.4225869770549129, 0.7467959387547628, 0.995, 0.34813070689980585],
            [0.44202386524503234, 0.754979882033474, 0.7596268745842244, -0.8369986991293447],
            [0.46121640273671693, 0.7382399080508871, 0.9548287402113034, -0.8328137056336979],
            [0.48121640273671695, 0.721583633938213, 0.995, -0.8286496371055294],
            [0.5011164027367173, 0.701583633938213, 0.990025, -0.995],
            [0.5209169027367174, 0.681583633938213, 0.985074875, -0.995],
            [0.5406184002367173, 0.661583633938213, 0.9801495006250001, -0.995],
            [0.5602213902492167, 0.6415836339382129, 0.9752487531218751, -0.995],
            [0.5757263653116537, 0.6216836339382136, 0.7713725093562657, -0.990025],
            [0.5871538154987784, 0.6018831339382135, 0.5685156468094843, -0.985074875],
            [0.5945241284349686, 0.5821816364382136, 0.36667306857543686, -0.9801495006250001],
            [0.5978575898064772, 0.5625786464257142, 0.16583970323255967, -0.9752487531218751],
            [0.5971743838711293, 0.5430736713632777, -0.033989495283603136, -0.9703725093562657],
            [0.6004945939654579, 0.5236662211761521, 0.1651804521928149, -0.9655206468094844],
            [0.5997982030093149, 0.5043558082399634, -0.034645450068149175, -0.960693043575437],
            [0.5991052940079519, 0.4851419473684545, -0.03447222281780843, -0.9558895783575597],
            [0.5984158495515953, 0.4660241558013031, -0.03429986170371939, -0.9511101304657719],
            [0.597729852317522, 0.44700195319198743, -0.03412836239520079, -0.946354579813443],
            [0.6010472850696188, 0.4280748615957183, 0.16504227941677524, -0.9416228069143757],
            [0.6083481306579549, 0.4092424054574303, 0.3632170680196914, -0.9369146928798039],
            [0.6156124720183483, 0.3905041115998339, 0.3614009826795929, -0.9322301194154049],
            [0.6268404916719391, 0.3718595092115259, 0.558593977766195, -0.9275689688183278],
            [0.6420123712272637, 0.3533081298351589, 0.754801007877364, -0.9229311239742362],
            [0.6611083913848111, 0.3348495073556742, 0.9500270028379773, -0.918316468354365],
            [0.6801089314415706, 0.32048317798858716, 0.9452768678237874, -0.7147248860125932],
            [0.6990144687980456, 0.3101886802683348, 0.9405504834846684, -0.5121512615825303],
            [0.7178254784677381, 0.30394565503668447, 0.9358477310672451, -0.3105905052746177],
            [0.7365424330890832, 0.2977338449311925, 0.9311684924119088, -0.30903755274824457],
            [0.7565424330890832, 0.2915530938762275, 0.995, -0.30749236498450333],
            [0.7764424330890826, 0.2854032465765374, 0.990025, -0.3059549031595808],
            [0.7962429330890827, 0.2792841485133454, 0.985074875, -0.3044251286437829],
            [0.8159444305890825, 0.2731956459404702, 0.9801495006250001, -0.302903003000564],
            [0.8355474206015819, 0.2631375858804593, 0.9752487531218751, -0.5003884879855612],
            [0.8550523956640185, 0.24912981612074853, 0.9703725093562657, -0.6968865455456335],
            [0.874459845851144, 0.23119208520983608, 0.9655206468094844, -0.8924021128179053],
            [0.8937702587873327, 0.21119208520983607, 0.960693043575437, -0.995],
            [0.9089841196588407, 0.19129208520983615, 0.7568895783575597, -0.990025],
            [0.9201219112259914, 0.17149158520983604, 0.5541051304657718, -0.985074875],
            [0.9272040138353063, 0.1517900877098362, 0.3523346048134429, -0.9801495006250001],
            [0.9302507059315753, 0.13218709769733622, 0.1515729317893757, -0.9752487531218751],
            [0.9332821645673625, 0.11668212263489869, 0.15081506713042883, -0.7713725093562657],
            [0.9322984659099717, 0.10125467244777325, -0.048939008205223324, -0.7675156468094844],
            [0.9313196857458669, 0.08990435951158354, -0.04869431316419721, -0.564678068575437],
            [0.9303457994825822, 0.08261079814007494, -0.04845084159837622, -0.3628546782325598],
            [0.9293767826506157, 0.07935370457542379, -0.04820858739038434, -0.16204040484139698],
        ]
        
        oracle_goals = [
            trajectory[12],
            trajectory[28],
            trajectory[44],
            # trajectory[51],
        ]

        for goal in oracle_goals:
            self.goals.append(goal[:2])
            self.goal_speeds.append(goal[2:])
        
        self.goals = np.array(self.goals)
        self.goal_speeds = np.array(self.goal_speeds)
        self.num_goals = self.goals.shape[0]

class PinballSuboptimalGoals(PinballGoals):
    def __init__(self):
        super().__init__()

        self.termination_radius = 0.04
        self.initiation_radius = 0.4
        self.speed_radius = 2.0
        
        # Caching calculation for functions
        self.termination_radius_squared = np.square(self.termination_radius)
        self.initiation_radius_squared = np.square(self.initiation_radius)
        self.speed_radius_squared = np.square(self.speed_radius)

        self.goals = self.goals[3, 5, 6, 9, 10]
        self.goal_speeds = self.goal_speeds[3, 5, 6, 9, 10]
        self.num_goals = self.goals.shape[0]

