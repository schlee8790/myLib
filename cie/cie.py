# cie.py
version="0.1.0"  # 240626


import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def XYZ_to_xy(XYZ: list):
    """Convert XYZ to xy"""
    if np.array(XYZ).shape[-1] == 3:
        X, Y, Z = XYZ[0], XYZ[1], XYZ[3]
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
    else:
        x, y = np.array([]), np.array([])
    return np.array([x, y])

def xy_from_XYZ(XYZ):
    return XYZ_to_xy(XYZ)


def XYZ_to_upvp(XYZ):
    if np.array(XYZ).shape[-1] == 3:
        X,Y,Z = XYZ[0], XYZ[1], XYZ[3]
        upvp_up = 4*X / (X + 15*Y + 3*Z)
        upvp_vp = 9*Y / (X + 15*Y + 3*Z)
    else:
        upvp_up, upvp_vp = np.array([]), np.array([])
    return np.array([upvp_up, upvp_vp])

def upvp_from_XYZ(XYZ):
    return XYZ_to_upvp(XYZ)


def xy_to_upvp(xy):
    if np.array(xy).shape[-1] == 2:
        x,y = xy[0], xy[1]
        upvp_up = 4*x / (-2*x + 12*y + 3)
        upvp_vp = 9*y / (-2*x + 12*y + 3)
    else:
        upvp_up, upvp_vp = np.array([]), np.array([])
    return np.array([upvp_up, upvp_vp])

def upvp_from_xy(xy):
    return xy_to_upvp(xy)


def XYZ_to_Lab(XYZ):
    Xn, Yn, Zn = 95.0489, 100.0, 108.8840
    if np.array(XYZ).shape[-1] == 3:
        X,Y,Z = np.array(XYZ[0]), np.array(XYZ[1]), np.array(XYZ[2])

        def f(t):
            return np.where(t > 0.008856451679, t**(1/3), t*7.787068965517+0.137931034483)

        X2, Y2, Z2 = X/Xn, Y/Yn, Z/Zn
        L = 116.0 * f(Y2) - 16.0
        a = 500.0 * (f(X2) - f(Y2))
        b = 200.0 * (f(Y2) - f(Z2))
    else:
        L, a, b = np.array([]), np.array([]), np.array([])
    return np.array([L, a, b])

def Lab_from_XYZ(XYZ):
    return XYZ_to_Lab(XYZ)


def Lab_to_dE00(Lab1: list, Lab2: list):
    """ Calculate the DE2000 from XYZs
    source: https://hajim.rochester.edu/ece/sites/gsharma/papers/CIEDE2000CRNAFeb05.pdf
    source: https://en.wikipedia.org/wiki/Color_difference
    RT = hue rotation term
    SL = compensation for lightness
    SC = compensation for chroma
    SH = compensation for hue
    This formula should use degrees rather than radians; the issue is significant for RT.
    """
    L1, a1, b1 = Lab1[0], Lab1[1], Lab1[2]
    L2, a2, b2 = Lab2[0], Lab2[1], Lab2[2]
    sqrt = np.sqrt
    # parametric weighting factors
    kL, kC, kH = 1., 1., 1.

    # 1. Calculate C'1, h'1
    C1, C2 = sqrt(a1**2 + b1**2), sqrt(a2**2 + b2**2)  # eq(2)
    C_ = (C1 + C2) / 2  # eq(3)
    G = 0.5 * (1 - sqrt(C_**7 / (C_**7 + 25**7)))  # eq(4)
    a1p, a2p = (1 + G) * a1, (1 + G) * a2  # eq(5)
    C1p, C2p = sqrt(a1p**2 + b1**2), sqrt(a2p**2 + b2**2)  # eq(6)
    if b1 == 0 and a1p == 0:  # eq(7)
        h1p = 0  # modified hue
    else:
        h1p = np.degrees(np.arctan2(b1, a1p) + np.pi * 2)  # degrees(0~360)

    if b2 == 0 and a2p == 0:  # eq(7)
        h2p = 0
    else:
        h2p = np.degrees(np.arctan2(b2, a2p) + np.pi * 2)  # degrees(0~360)

    # 2. Calculate dL', dC', dH'
    dLp = L2 - L1  # eq(8)
    dCp = C2p - C1p  # eq(9)
    dhp = 0.0
    if C1p * C2p == 0:  # eq(10)
        dhp = 0.0
    elif C1p * C2p != 0 and abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif C1p * C2p != 0 and (h2p - h1p) > 180:
        dhp = (h2p - h1p) - 360
    elif C1p * C2p != 0 and (h2p - h1p) < -180:
        dhp = (h2p - h1p) + 360

    dHp = 2 * sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))  # eq(11)

    # 3. Calculate CIEDE2000 Color-Difference dE00
    L_p = (L1 + L2) / 2  # eq(12)
    C_p = (C1p + C2p) / 2  # eq(13)
    h_p = 0
    if abs(h1p - h2p) <= 180 and C1p * C2p != 0:  # eq(14)
        h_p = (h1p + h2p) / 2
    elif abs(h1p - h2p) > 180 and (h1p + h2p) < 360 and C1p * C2p != 0:
        h_p = (h1p + h2p + 360) / 2
    elif abs(h1p - h2p) > 180 and (h1p + h2p) >= 360 and C1p * C2p != 0:
        h_p = (h1p + h2p - 360) / 2
    elif C1p * C2p == 0:
        h_p = h1p + h2p

    T = (1 - 0.17 * np.cos(np.radians(h_p - 30)) + 0.24 * np.cos(np.radians(2 * h_p))
        + 0.32 * np.cos(np.radians(3 * h_p + 6)) - 0.20 * np.cos(np.radians(4 * h_p - 63))
    )  # eq(15)
    dtheta = 30 * np.exp(-(((h_p - 275) / 25) ** 2))  # eq(16)
    RC = 2 * sqrt(C_p**7 / (C_p**7 + 25**7))  # eq(17)
    SL = 1 + (0.015 * (L_p - 50)**2 / sqrt(20 + (L_p - 50)**2))  # eq(18)
    SC = 1 + 0.045 * C_p  # eq(19)
    SH = 1 + 0.015 * C_p * T  # eq(20)
    RT = -np.sin(np.radians(2 * dtheta)) * RC  # eq(21)

    dE00 = sqrt((dLp / (kL * SL))**2 + (dCp / (kC * SC))**2 + (dHp / (kH * SH))**2
        + RT * (dCp / (kC * SC)) * (dHp / (kH * SH))
    )  # eq(22)

    return dE00

def dE00_from_Lab(Lab1, Lab2):
    return Lab_to_dE00(Lab1, Lab2)



class HorseShoe_xy1931:
    def __init__(self):
        self._wd = np.array([
            390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,
            500,505,510,515,520,525,530,535,540,545,550,555,560,565,570,575,580,585,590,595,600,605,
            610,615,620,625,630,635,640,645,650,655,660,665,670,675,680,685,690,700])
        self._xy_x = np.array([
            0.173336886,0.173020965,0.172576551,0.172086631,0.171407434,0.170301002,0.168877521,0.16689529,
            0.164411756,0.16110458,0.156640933,0.150985408,0.143960396,0.135502671,0.124118477,0.109594324,
            0.091293516,0.06870591,0.045390735,0.023459943,0.008168028,0.003858521,0.013870246,0.038851802,
            0.074302424,0.114160721,0.154722061,0.192876183,0.229619673,0.265775085,0.301603869,0.337363289,
            0.373101544,0.408736255,0.444062464,0.478774791,0.512486367,0.544786506,0.575151311,0.602932786,
            0.6270366,0.648233106,0.665763576,0.68007885,0.691503998,0.700606061,0.707917792,0.714031597,
            0.719032942,0.723031603,0.725992318,0.728271728,0.729969013,0.731089396,0.7319933,0.732718894,
            0.733416967,0.7340473,0.734390165,0.734591662,0.734687278,0.173336886])
        self._xy_y = np.array([
            0.004796744,0.00477505,0.004799302,0.004832524,0.005102171,0.005788505,0.006900244,0.008555606,
            0.010857558,0.013793359,0.017704805,0.022740193,0.02970297,0.039879121,0.057802513,0.086842511,
            0.132702055,0.20072322,0.294975965,0.412703479,0.538423071,0.654823151,0.750186428,0.812016021,
            0.833803082,0.826206968,0.805863545,0.781629131,0.75432909,0.724323925,0.692307692,0.658848333,
            0.62445086,0.589606869,0.554713903,0.520202307,0.486590788,0.454434115,0.424232235,0.396496634,
            0.372491145,0.351394916,0.334010651,0.319747217,0.308342236,0.299300699,0.292027109,0.285928874,
            0.280934952,0.276948358,0.274007682,0.271728272,0.270030987,0.268910604,0.2680067,0.267281106,
            0.266583033,0.2659527,0.265609835,0.265408338,0.265312722,0.004796744])

    @property
    def wd(self):
        return self._wd
    @property
    def xy_x(self):
        return self._xy_x
    @property
    def xy_y(self):
        return self._xy_y

    def is_inside(self, point):
        polygon = np.dstack([self._xy_x, self._xy_y])
        num_vertices = len(polygon)
        intersection_count = 0
        for i in range(num_vertices):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % num_vertices]
            if (y1 > point[1]) != (y2 > point[1]):
                if point[0] < (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1:
                    intersection_count += 1
        return intersection_count % 2 == 1  # True, False


class xy1931_horseshoe(HorseShoe_xy1931):
    def __init__(self):
        super().__init__()



class Plankian_xy1931:
    def __init__(self):
        self._cct = np.array([
            2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5200,5400,5600,5800,6000,
            6200,6600,7000,7400,7800,8200,8600,9000,9400,9800,10000,11000,12000,13000,14000])
        self._xy_x = np.array([
            0.4861184,0.4682142,0.4518347,0.4369156,0.4233474,0.4110268,0.3998348,0.3896738,0.3804363,0.3720269,
            0.3643663,0.3573696,0.3509675,0.3451047,0.3397208,0.3347661,0.3302041,0.3259905,0.322091,0.318479,
            0.3120048,0.306387,0.3014791,0.2971652,0.2933537,0.2899672,0.2869411,0.284229,0.2817827,0.2806493,
            0.2757343,0.2718017,0.2686017,0.2659523])
        self._xy_y = np.array([
            0.4146904,0.4123223,0.4086497,0.4040903,0.3989732,0.3935257,0.3879313,0.3823166,0.3767725,0.371358,
            0.3661172,0.36108,0.3562481,0.3516441,0.3472518,0.3430781,0.3391196,0.3353579,0.331791,0.328415,
            0.322174,0.3165618,0.311504,0.3069374,0.3028035,0.2990557,0.295643,0.2925295,0.2896825,0.2883483,
            0.2824639,0.2776349,0.2736156,0.2702374])
    @property
    def cct(self):
        return self._cct
    @property
    def xy_x(self):
        return self._xy_x
    @property
    def xy_y(self):
        return self._xy_y


class xy1931_plankian(Plankian_xy1931):
    def __init__(self):
        super().__init__()



class HorseShoe_uv1976:
    def __init__(self):
        self._wd = np.array([
            390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,500,
            505,510,515,520,525,530,535,540,545,550,555,560,565,570,575,580,585,590,595,600,605,610,615,
            620,625,630,635,640,645,650,655,660,665,670,675,680,685,690,700])
        self._upvp_up = np.array([
            0.255764075,0.255262941,0.254496534,0.253645148,0.252217082,0.249629513,0.246083169,0.241101802,
            0.234750929,0.226643616,0.216117881,0.203284963,0.187661332,0.168979963,0.144097895,0.114670755,
            0.082808959,0.052136166,0.028153963,0.011870155,0.003459292,0.001422475,0.004633262,0.012269163,
            0.023116509,0.035995353,0.050068144,0.064325311,0.079228991,0.095257011,0.112701874,0.131892616,
            0.153111157,0.176601706,0.202573031,0.231155986,0.262338731,0.29593341,0.331476188,0.368085551,
            0.403510105,0.437975161,0.469128351,0.496697154,0.520211506,0.539924911,0.556485582,0.570873244,
            0.583020931,0.592974859,0.600476569,0.606363069,0.610797296,0.613748576,0.616143814,0.618075802,
            0.61994234,0.621634388,0.622557422,0.623100752,0.623358806,0.255764075])
        self._upvp_vp = np.array([
            0.015924933,0.015850739,0.015924312,0.0160264,0.01689205,0.019090957,0.022623356,0.027809276,
            0.034881017,0.043660445,0.054961567,0.068888535,0.087119257,0.11189623,0.150990784,0.204446417,
            0.270830492,0.342708696,0.411662654,0.469840116,0.513069415,0.543163229,0.563838135,0.576966861,
            0.583667184,0.586139654,0.586750245,0.586525023,0.585623376,0.584114563,0.582070925,0.579549626,
            0.576581333,0.573187597,0.569362856,0.565104363,0.560436611,0.555419473,0.550118696,0.544630417,
            0.539336727,0.53419063,0.529559157,0.525438262,0.521916219,0.518978928,0.516507672,0.514354792,
            0.512535145,0.511046375,0.509928515,0.50904554,0.508380406,0.507937714,0.507578428,0.50728863,
            0.507008649,0.506754842,0.506616387,0.506534887,0.506496179,0.015924933])

    @property
    def wd(self):
        return self._wd
    @property
    def upvp_up(self):
        return self._upvp_up
    @property
    def upvp_vp(self):
        return self._upvp_vp

    def is_inside(self, point):
        polygon = np.dstack([self._upvp_up, self._upvp_vp])
        num_vertices = len(polygon)
        intersection_count = 0
        for i in range(num_vertices):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % num_vertices]
            if (y1 > point[1]) != (y2 > point[1]):
                if point[0] < (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1:
                    intersection_count += 1
        return intersection_count % 2 == 1  # True, False


class uv1976_horseshoe(HorseShoe_uv1976):
    def __init__(self):
        super().__init__()



class Plankian_uv1976():
    def __init__(self):
        self._cct = np.array([
            2300,2350,2400,2450,2500,2550,2600,2650,2700,2750,2800,2850,2900,2950,3000,3050,3100,3150,3200,
            3250,3300,3350,3400,3450,3500,3550,3600,3650,3700,3750,3800,3850,3900,3950,4000,4050,4100,4150,
            4200,4250,4300,4350,4400,4450,4500,4550,4600,4650,4700,4750,4800,4850,4900,4950,5000,5050,5100,
            5150,5200,5250,5300,5350,5400,5450,5500,5550,5600,5650,5700,5750,5800,5850,5900,5950,6000,6050,
            6100,6150,6200,6250,6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,7600,7700,
            7800,7900,8000,8100,8200,8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,
            9700,9800,9900,10000,10500,11000,11500,12000,12500,13000,13500,14000])
        self._upvp_up = np.array([
            0.283579934,0.280536184,0.277621398,0.274827773,0.272148729,0.269580532,0.267114461,0.264746878,
            0.26247381,0.260288068,0.258186573,0.256168743,0.254223892,0.252352319,0.250551851,0.24881502,
            0.247138928,0.245527515,0.243969688,0.242467484,0.241015746,0.239619255,0.238267607,0.236958665,
            0.235698235,0.234474619,0.233292656,0.232150463,0.231044067,0.229971824,0.228935343,0.227931031,
            0.226957089,0.226013009,0.225097004,0.22420842,0.223348245,0.222514141,0.221700522,0.220913561,
            0.22014826,0.219407438,0.218685152,0.217981069,0.217301157,0.216636301,0.215991343,0.215362887,
            0.21475466,0.214160023,0.213579946,0.213015773,0.212469992,0.211934454,0.211411994,0.210906386,
            0.21041235,0.209930309,0.209459182,0.20900185,0.208556009,0.208116991,0.207690443,0.207278337,
            0.206872579,0.206475894,0.206086883,0.205709896,0.205342338,0.204982559,0.204629282,0.204285521,
            0.203948629,0.20361917,0.203298245,0.202983343,0.202675504,0.202374657,0.202079879,0.201792133,
            0.2015097,0.200963452,0.20043867,0.199936483,0.199453318,0.19899166,0.198545912,0.19811743,0.197705365,
            0.197309318,0.196926905,0.196560513,0.196205689,0.195863632,0.195535587,0.195217068,0.194909914,
            0.194612912,0.194327823,0.194051181,0.19378397,0.193522784,0.193273034,0.193030479,0.192795881,
            0.192567133,0.192347136,0.192131961,0.191926164,0.191723177,0.191527751,0.191338926,0.19115412,
            0.190973971,0.190800768,0.190631213,0.190467537,0.190306806,0.189568798,0.188920632,0.188346664,
            0.187837579,0.18738456,0.186977451,0.186610263,0.186275537])
        self._upvp_vp = np.array([
            0.534519892,0.533706867,0.532865223,0.531996398,0.531105281,0.53019331,0.529263764,0.52831619,
            0.527354783,0.526384303,0.525397225,0.524406036,0.52340406,0.52239956,0.521387972,0.520375918,
            0.519360459,0.518342543,0.517327083,0.516310399,0.5152971,0.514281446,0.513275437,0.51226766,
            0.511269034,0.510271704,0.509281401,0.508294902,0.507316762,0.506343934,0.505379138,0.50441907,
            0.503470019,0.502525079,0.501590705,0.500663214,0.499744964,0.498832388,0.497929291,0.497037648,
            0.496151719,0.4952729,0.494406013,0.4935481,0.492698331,0.49185587,0.491026229,0.490201261,
            0.4893892,0.488581577,0.48778523,0.486999633,0.486218586,0.485448917,0.484690605,0.483936918,
            0.483192361,0.482456689,0.481730661,0.481013229,0.480302194,0.479599923,0.478906304,0.478222175,
            0.477543119,0.476876536,0.476215251,0.475559341,0.474914867,0.474278314,0.473646025,0.473021942,
            0.472408282,0.471799284,0.471196612,0.470605277,0.470017118,0.469438105,0.468864956,0.468298783,
            0.467738831,0.466639881,0.465567377,0.464519318,0.463496075,0.46249556,0.461521461,0.460567592,
            0.459637037,0.458725063,0.457834825,0.456967394,0.456117107,0.455287954,0.454475526,0.453682644,
            0.452906762,0.452148065,0.45140695,0.450679837,0.449969155,0.449275239,0.448594935,0.447931511,
            0.447277679,0.446642008,0.446016642,0.445406934,0.444807435,0.444222438,0.443647316,0.443085086,
            0.442535531,0.441993752,0.441464769,0.440945042,0.440437558,0.439936778,0.437584969,0.435445759,
            0.433491291,0.431704812,0.43006276,0.42855232,0.427161889,0.425872941])

    @property
    def cct(self):
        return self._cct
    @property
    def upvp_up(self):
        return self._upvp_up
    @property
    def upvp_vp(self):
        return self._upvp_vp


class uv1976_plankian(Plankian_uv1976):
    def __init__(self):
        super().__init__()



class DominantWavelength():
    def __init__(self):
        self._wd = np.array([
            360,365,370,375,380,385,390,395,400,405,410,415,420,425,430,435,440,445,450,455,460,465,470,475,480,485,490,495,500,
            505,510,515,520,525,530,535,540,545,550,555,560,565,570,575,580,585,590,595,600,605,610,615,620,625,630,635,640,645,
            650,655,660,665,670,675,680,685,690,695,700,705,710,715,720,725,730,735,740,745,750,755,760,765,770,775,780,785,790,
            795,800,805,810,815,820,825,830])
        self._atan2deg = np.array([
            -154.3144167,-154.2604057,-154.2146045,-154.1732735,-154.1320055,-154.1161176,-154.0913345,-154.0568107,-154.0340378,
            -153.9910149,-153.9267818,-153.8555947,-153.7415474,-153.5386078,-153.2611995,-152.8663333,-152.3532306,-151.6757901,
            -150.7594072,-149.5829842,-148.0483894,-146.0143213,-142.7900297,-137.7700647,-129.6559641,-116.6163698,-97.58780477,
            -75.63327873,-57.75936398,-45.70277789,-37.46541907,-31.59947702,-27.36497888,-23.97391041,-20.706046,-17.3964878,
            -13.83942115,-9.803193952,-5.051212351,0.709307427,7.77876174,16.3953745,26.5730936,37.8938771,49.45448289,60.19991678,
            69.39885829,76.81423863,82.40585956,86.71730509,89.88326155,92.24380447,93.99129883,95.29408737,96.29270326,97.09792277,
            97.736431,98.23292322,98.59166353,98.86592141,99.06782953,99.20004276,99.3060969,99.39083656,99.47203313,99.54507464,
            99.58469554,99.6079443,99.61928357,99.61928207,99.61927953,99.61928253,99.61928138,99.61928092,99.61927538,99.61927999,
            99.61928161,99.61928092,99.61928218,99.61928115,99.61927538,99.61927158,99.6192808,99.61927919,99.61927907,99.61927919,
            99.61927562,99.61927804,99.61927953,99.61927988,99.61928069,99.61928195,99.61927907,99.61927746,99.61927619])
        # self._wdfunc = interp1d(self._atan2deg, self._wd)  # Scipy를 사용하는 경우
        self._wdfunc = np.interp(self._atan2deg, self._atan2deg, self._wd)  # Numpy를 사용하는 경우


    def wd(self):
        return self._wd


    def atan2deg(self):
        return self._atan2deg


    def xy_to_wavelength(self, x, y):
        x, y = np.array(x), np.array(y)
        _shape = x.shape

        xw, yw = 1/3, 1/3  # 원점
        dx = x.flatten()-xw if x.size > 1 else x-xw
        dy = y.flatten()-yw if y.size > 1 else y-yw  # 거리
        dxy = np.sqrt(dx**2 + dy**2)  # distance from 원점

        deg = np.degrees(np.arctan2(dx, dy))  # 방사각

        # 마젠타 파장은 purplish blue, purplish red 로 치환
        # magenta = np.where(deg < self._atan2deg[0], -1, 1) * np.where(deg > self._atan2deg[-1], -1, 1)
        deg = np.where(deg < self._atan2deg[0], self._atan2deg[0], deg)  # Purplish Blue: 360nm 지정
        deg = np.where(deg > self._atan2deg[-1], self._atan2deg[-1], deg)  # Purplish Red: 830nm 지정

        # deg_magenta = np.where(magenta>0, deg, np.degrees(np.arctan2(-dx, -dy)))  # magenta 파장은 상보색(-)으로 대체
        # wd_magenta = np.interp(deg_magenta, self._atan2deg, self._wdfunc)  # magenta를 상보색으로 처리
        wd = np.interp(deg, self._atan2deg, self._wdfunc)  # magenta를 딥블루,딥레드로 처리
        # wd = wd_magenta * magenta

        # wd = np.where(dxy < 0.1, self._wd[0], wd)  # White부근은 360nm로 고정
        wd = np.where(dxy < 0.001, 0, wd)  # White부근은 중심파장 0으로 고정
        wd_reshaped = wd.reshape(_shape)
        return wd_reshaped



def wavelength_to_rgb(wavelength, gamma):
    ''' 가시광선 영역의 파장에 해당하는 RGB 색상을 [0~255] 범위의 RGB 값으로 반환
    Parameter:
        wavelength: 파장(nm)
        gamma
    Return:
        uint8 형식의 (R,G,B): [0~255]
    '''
    # gamma = 1.5
    intensity_max = 255

    if 380 <= wavelength < 440:
        a = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = (-(wavelength - 440) / (440 - 380)*a)**gamma
        G = 0.0
        B = a**gamma
    elif 440 <= wavelength < 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440))**gamma
        B = 1.0
    elif 490 <= wavelength < 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490))*gamma
    elif 510 <= wavelength < 580:
        R =((wavelength - 510) / (580 - 510))**gamma
        G = 1.0
        B = 0.0
    elif 580 <= wavelength < 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580))**gamma
        B = 0.0
    elif 645 <= wavelength <= 780:
        a = 0.3 + 0.7 * (780-wavelength) / (780 - 645)
        R = (1.0*a)**gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    R = int(R * intensity_max)
    G = int(G * intensity_max)
    B = int(B * intensity_max)

    return (R, G, B)



class Colormap_Wavelength():
    def __init__(self):
        ''' 가시광선 영역의 파장을 색상으로 표시하는 칼라맵
        '''
        self._gamma = 1.0
        self._wavelengths = np.array([])
        self._wave_start, self._wave_end, self._wave_step = 380., 780., 0.2
        self._colors = []
        self._cmap = []

    def set_gamma(self, value: float):
        ''' 감마값 지정(default=1.0)
        '''
        self._gamma = value


    def set_wavelengths(self, wstart, wend, wstep):
        ''' 파장 범위 지정
        Params: (기본값: 380~780nm)
        '''
        self._wave_start, self._wave_end = wstart, wend
        if wstep is None:
            self._wave_step = wstep = 0.2
        self._wavelengths = np.arange(wstart, wend + wstep, wstep)


    def colormap(self, waves=None, gamma=None) -> ListedColormap:
        wstep = self._wave_step
        if gamma is None:
            gamma = self._gamma
        if waves is not None:
            self._wavelengths = np.arange(waves[0], waves[1] + wstep, wstep)
        else:
            self._wavelengths = np.arange(self._wave_start, self._wave_end + self._wave_step, self._wave_step)
        self._colors = [self.wavelength_to_rgb(w, gamma) for w in self._wavelengths]
        self._cmap = ListedColormap(np.array(self._colors))
        return self._cmap


    # 파장값을 RGB 색상으로 변환합니다.
    def wavelength_to_rgb(self, wavelength, gamma) -> tuple:
        ''' 가시광선 영역의 파장에 해당하는 RGB 색상을 [0~1] 범위의 RGB 값으로 반환
        Parameter:
            wavelength: 파장(nm)
            gamma
        Return:
            [0~1] 범위의 (R,G,B)
        '''
        if 380 <= wavelength < 440:
            a = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
            R = (-(wavelength - 440) / (440 - 380)*a)**gamma
            G = 0.0
            B = a**gamma
        elif 440 <= wavelength < 490:
            R = 0.0
            G = ((wavelength - 440) / (490 - 440))**gamma
            B = 1.0
        elif 490 <= wavelength < 510:
            R = 0.0
            G = 1.0
            B = (-(wavelength - 510) / (510 - 490))*gamma
        elif 510 <= wavelength < 580:
            R =((wavelength - 510) / (580 - 510))**gamma
            G = 1.0
            B = 0.0
        elif 580 <= wavelength < 645:
            R = 1.0
            G = (-(wavelength - 645) / (645 - 580))**gamma
            B = 0.0
        elif 645 <= wavelength <= 780:
            a = 0.3 + 0.7 * (780-wavelength) / (780 - 645)
            R = (1.0*a)**gamma
            G = 0.0
            B = 0.0
        else:
            R, G, B = 0.0, 0.0, 0.0

        R = 1 if R > 1 else R
        G = 1 if G > 1 else G
        B = 1 if B > 1 else B
        return (R, G, B)


class Colormap_Excel:
    """ 엑셀의 색조맵 (색순서: low_mid_high)

    Usage:
        cmap = Colormap_Excel().bwr()  # blue-white-red
    """
    def __init__(self):
        a = 255
        self.colors = {
            'bwr': [(90/a, 138/a, 198/a), (252/a, 252/a, 255/a), (248/a, 105/a, 107/a)],  }
        self.colors['bwr_r'] = list(reversed(self.colors['bwr']))

    def _colormap(self, color: str)  -> LinearSegmentedColormap:
        return LinearSegmentedColormap.from_list('custom_cmap', self.colors[color])

    def bwr(self)  -> LinearSegmentedColormap:
        """ Blue-White-Red """
        return self._colormap('bwr')

    def bwr_r(self)  -> LinearSegmentedColormap:
        """ Red-White-Blue """
        return self._colormap('bwr_r')

    def rwb(self)  -> LinearSegmentedColormap:
        """ Red-White-Blue """
        return self._colormap('bwr_r')

    # 호출시 에러 발생하는 원인 확인후 사용 결정할 예정...당장은 보류함(240512)
    # def __call__(self)  -> LinearSegmentedColormap:
    #     return self._cmap


# class Colormap_BWR_Excel:
#     """ 엑셀의 blue_white_red 색조맵 (low_mid_high)

#     Usage:
#         cmap = Colormap_BlueWhiteRed().colormap()
#     """
#     def __init__(self):
#         a = 255
#         excel_colors = [(90/a, 138/a, 198/a), (252/a, 252/a, 255/a), (248/a, 105/a, 107/a)]
#         self._cmap = LinearSegmentedColormap.from_list('custom_cmap', excel_colors)

#     def colormap(self)  -> LinearSegmentedColormap:
#         return self._cmap



class Luminous_Efficiency:
    ''' 파장별 밝기 민감도
    '''
    def __init__(self):
        self._wave = np.arange(390, 830+1, 5)
        self._luminous_2deg = np.array([
            0.000414616,0.00105965,0.00245219,0.00497172,0.00907986,0.0142938,0.0202737,0.0261211,0.0331904,0.0415794,0.0503366,
            0.0574339,0.0647235,0.0723834,0.0851482,0.106014,0.129896,0.153507,0.178805,0.206483,0.237916,0.285068,0.348354,
            0.42776,0.520497,0.620626,0.718089,0.794645,0.85758,0.907135,0.954468,0.981411,0.989023,0.999461,0.996774,0.990255,
            0.973261,0.942457,0.896361,0.85872,0.811587,0.754479,0.691855,0.627007,0.558375,0.489595,0.42299,0.360924,0.298086,
            0.24169,0.194312,0.15474,0.119312,0.0897959,0.0667104,0.048997,0.0355998,0.0255422,0.0180794,0.0126157,0.00866128,
            0.00602768,0.00419594,0.00291086,0.00199556,0.00136702,0.000944727,0.000653705,0.000455597,0.000317974,0.000221745,
            0.000156557,0.000110393,7.82744E-05,5.57886E-05,3.98188E-05,2.86018E-05,2.05126E-05,1.48724E-05,0.0000108,7.86392E-06,
            5.73694E-06,4.2116E-06,3.10656E-06,2.28679E-06,1.69315E-06,1.26256E-06,9.42251E-07,7.05386E-07 ])
        self._luminous_func_2deg = np.interp(self._wave, self._wave, self._luminous_2deg)
        self._luminous_10deg = np.array([
            0.000407678, 0.00107817, 0.00258977, 0.00547421, 0.010413, 0.0171297, 0.0257613, 0.0352955, 0.0469823, 0.0604743,
            0.0746829, 0.0882054, 0.103903, 0.119539, 0.141459, 0.170137, 0.199986, 0.231243, 0.268227, 0.310944, 0.355402,
            0.414823, 0.478048, 0.549134, 0.62483, 0.701229, 0.77882, 0.837636, 0.882955, 0.923386, 0.966532, 0.988689,
            0.99075, 0.999778, 0.99443, 0.984813, 0.964055, 0.928649, 0.877536, 0.837084, 0.786995, 0.727231, 0.662904,
            0.597037, 0.52823, 0.460131, 0.395076, 0.335179, 0.275181, 0.221956, 0.177688, 0.14102, 0.1084, 0.0813769,
            0.0603398, 0.0442538, 0.0321185, 0.0230257, 0.0162884, 0.0113611, 0.00779746, 0.00542539, 0.00377614, 0.00261937,
            0.0017956, 0.00122998, 0.00084999, 0.000588138, 0.000409893, 0.000286072, 0.000199495, 0.000140847, 0.0000993144,
            0.0000704188, 0.0000501893, 0.0000358222, 0.0000257308, 0.0000184535, 0.0000133795, 0.0000097158, 0.00000707442,
            0.00000516095, 0.00000378873, 0.00000279463, 0.00000205715, 0.00000152311, 0.00000113576, 0.000000847617, 0.000000634538 ])
        self._luminous_func_10deg = np.interp(self._wave, self._wave, self._luminous_10deg)


    def get(self, wave, fov: str="2deg"):
        ''' Luminous Efficiency Function

            Args:
                wave (float): wavelength(nm)
                fov (str): field of viewing (2deg or 10deg, default=2deg if fov="")
            Returns:
                luminous efficiency at given wavelength and fov
        '''
        wave = np.array(wave)
        _shape = wave.shape
        if fov == "10deg":
            efficiency = np.interp(wave.flatten(), self._wave, self._luminous_func_10deg)
        else:
            efficiency = np.interp(wave.flatten(), self._wave, self._luminous_func_2deg)
        efficiency_reshaped = efficiency.reshape(_shape)
        return efficiency_reshaped



class Wavelength_JND_by_pokorny_1970:
    """ Wavelength_Discrimination_Threshold by Pokorny and Smith(1970)
        References:
            [1] Joel Pokorny and Vivianne C. Smith,
                "Wavelength Discrimination in the Presence of Added Chromatic Fields",
                Journal Of The Optical Society Of America, Vol. 60, Number 4, 1970
    """

    def __init__(self):
        self._wave = np.arange(440, 670+1, 10)
        # self._jnd = np.array([1.5, 4.5, 3.0, 2.0, 1.5, 1.0, 1.5, 2.5, 3.0, 3.5, 3.0, 2.5,
        #                       2.0, 2.0, 1.5, 1.0, 1.0, 1.5, 2.0, 2.5, 3.0, 5.5, 8.0, 9.0 ])
        self._jnd = np.array([3.75, 3.25, 2.125, 1.375, 1.125, 0.875, 1.0, 1.625, 2.375,
                              2.875, 3.0, 2.75, 2.0, 1.5, 1.0, 0.875, 1.375, 1.875, 2.875,
                              3.875, 5.0, 6.667, 8.0, 9.0, ])
        self._jnd_func = np.interp(self._wave, self._wave, self._jnd)


    def get(self, wave):
        """ Wavelength Discrimination

            Args:
                wave (float): wavelength(nm)
            Returns:
                luminous efficiency at given wavelength
        """
        wave = np.array(wave)
        _shape = wave.shape
        jnd = np.interp(wave.flatten(), self._wave, self._jnd_func)
        jnd_reshaped = jnd.reshape(_shape)
        return jnd_reshaped



class Wavelength_JND_by_zhaoping_2011:
    """ Wavelength_Discrimination_Threshold by Zhaoping(2011)
        References:
        [1] Zhaoping L, Geisler WS, May KA (2011)
            "Human Wavelength Discrimination of Monochromatic Light Explained
             by Optimal Wavelength Decoding of Light of Unknown Intensity",
             PLoS ONE 6(5): e19248.
             doi:10.1371/journal.pone.0019248
    """

    def __init__(self):
        self._wave = np.arange(415, 670+1, 5)
        self._jnd = np.array([
            25, 12, 7.5, 5, 3.7, 3.2, 2.8, 2.5, 2.3, 2, 1.6, 1.3, 1.2, 1.2, 1.25,
            1.32, 1.5, 1.7, 1.85, 1.95, 2.15, 2.45, 2.7, 2.8, 2.75, 2.65, 2.5,
            2.3, 2.1, 1.9, 1.7, 1.6, 1.4, 1.3, 1.2, 1.15, 1.15, 1.18, 1.22,
            1.35, 1.55, 1.8, 2.16, 2.6, 3.1, 4, 5.2, 7.5, 11, 17, 25, 35, ])
        self._jnd_func = np.interp(self._wave, self._wave, self._jnd)


    def get(self, wave):
        """ Wavelength discrimination threshold(JND) at a given wavelength

            Args:
                wave (float): wavelength(nm)
            Returns:
                JND at given wavelength
        """
        wave = np.array(wave)
        _shape = wave.shape
        jnd = np.interp(wave.flatten(), self._wave, self._jnd_func)
        jnd_reshaped = jnd.reshape(_shape)
        return jnd_reshaped



