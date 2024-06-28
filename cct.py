# cct.py
version="0.1.0"  # 240626

"""
Correlated Color Temperture

"""

import math
import numpy as np

# LERP(a,b,c) = linear interpolation macro, is 'a' when c == 0.0 and 'b' when c == 1.0
def LERP(a, b, c):
    return ((b - a) * c + a)

class UVT:
    def __init__(self, u, v, t):
        self.u = u  # uv1960-u,v
        self.v = v
        self.t = t

uvt = [
    UVT(0.18006, 0.26352, -0.24341),
    UVT(0.18066, 0.26589, -0.25479),
    UVT(0.18133, 0.26846, -0.26876),
    UVT(0.18208, 0.27119, -0.28539),
    UVT(0.18293, 0.27407, -0.30470),
    UVT(0.18388, 0.27709, -0.32675),
    UVT(0.18494, 0.28021, -0.35156),
    UVT(0.18611, 0.28342, -0.37915),
    UVT(0.18740, 0.28668, -0.40955),
    UVT(0.18880, 0.28997, -0.44278),
    UVT(0.19032, 0.29326, -0.47888),
    UVT(0.19462, 0.30141, -0.58204),
    UVT(0.19962, 0.30921, -0.70471),
    UVT(0.20525, 0.31647, -0.84901),
    UVT(0.21142, 0.32312, -1.0182),
    UVT(0.21807, 0.32909, -1.2168),
    UVT(0.22511, 0.33439, -1.4512),
    UVT(0.23247, 0.33904, -1.7298),
    UVT(0.24010, 0.34308, -2.0637),
    UVT(0.24792, 0.34655, -2.4681),  # Note: 0.24792 is a corrected value for the error found in W&S as 0.24702
    UVT(0.25591, 0.34951, -2.9641),
    UVT(0.26400, 0.35200, -3.5814),
    UVT(0.27218, 0.35407, -4.3633),
    UVT(0.28039, 0.35577, -5.3762),
    UVT(0.28863, 0.35714, -6.7262),
    UVT(0.29685, 0.35823, -8.5955),
    UVT(0.30505, 0.35907, -11.324),
    UVT(0.31320, 0.35968, -15.628),
    UVT(0.32129, 0.36011, -23.325),
    UVT(0.32931, 0.36038, -40.770),
    UVT(0.33724, 0.36051, -116.45)
]

rt = [
    math.ldexp(1.0, -1022), 10.0e-6, 20.0e-6, 30.0e-6, 40.0e-6, 50.0e-6,
    60.0e-6, 70.0e-6, 80.0e-6, 90.0e-6, 100.0e-6, 125.0e-6,
    150.0e-6, 175.0e-6, 200.0e-6, 225.0e-6, 250.0e-6, 275.0e-6,
    300.0e-6, 325.0e-6, 350.0e-6, 375.0e-6, 400.0e-6, 425.0e-6,
    450.0e-6, 475.0e-6, 500.0e-6, 525.0e-6, 550.0e-6, 575.0e-6,
    600.0e-6
]

# from Wikipedia
# https://en.wikipedia.org/wiki/Correlated_color_temperature
def xy_to_cct(x, y):
    xe, ye = 0.3366, 0.1735
    a0, a1, a2, a3 = -949.86315, 6253.80338, 28.70599, 0.00004
    t1, t2, t3 = 0.92159, 0.20039, 0.07125
    n = (x - xe) / (y - ye)
    cct = a0 + a1 * np.exp(-n / t1) + a2 * np.exp(-n / t2) + a3 * np.exp(-n / t3)
    return cct


def xy_to_uv(x, y):
    denom = -2*x + 12*y + 3
    us = (4.0 * x) / denom  # uv1960-u
    vs = (6.0 * y) / denom  # uv1960-v
    return us, vs


def uv_to_xy(u, v):  # uv1960-u,v
    denom = 2*u -8*v +4
    x = 3*u / denom
    y = 2*v / denom
    return x, y


'''def xy_to_uv(xy):
    denom = -2*xy[0] + 12*xy[1] + 3
    us = (4.0 * xy[0]) / denom  # uv1960-u
    vs = (6.0 * xy[1]) / denom  # uv1960-v
    return us, vs
'''

'''def uv_to_xy(uv):
    us,vs = uv[0], uv[1]
    denom = 2*us -8*vs +4
    x = 3*us / denom
    y = 2*vs / denom
    return x, y
'''

'''def XYZ_to_CCT(XYZ):
    sqrt = math.sqrt
    denom = XYZ[0] + 15.0 * XYZ[1] + 3.0 * XYZ[2]
    us = (4.0 * XYZ[0]) / denom  # uv1960-u
    vs = (6.0 * XYZ[1]) / denom  # uv1960-v
    if us < 0.14 or us > 0.32 or vs < 0.25 or vs > 0.38:  # us,vs points are not white area.
        return -1
    dm = 0.0
    for i in range(31):
        di = (vs - uvt[i].v) - uvt[i].t * (us - uvt[i].u)
        if i > 0 and ((di < 0.0 and dm >= 0.0) or (di >= 0.0 and dm < 0.0)):
            break  # found lines bounding (us, vs) : i-1 and i
        dm = di
    if i == 31:
        return -1  # bad XYZ input, color temp would be less than minimum of 1666.7 degrees, or too far towards blue
    di = di / sqrt(1.0 + uvt[i].t * uvt[i].t)
    dm = dm / sqrt(1.0 + uvt[i - 1].t * uvt[i - 1].t)
    p = dm / (dm - di)  # p = interpolation parameter, 0.0 : i-1, 1.0 : i
    p = 1.0 / (LERP(rt[i - 1], rt[i], p))
    temp = p
    return temp
'''

'''def xy_to_cct(xy):
    sqrt = math.sqrt
    denom = -2*xy[0] + 12*xy[1] + 3
    us = (4.0 * xy[0]) / denom  # uv1960-u
    vs = (6.0 * xy[1]) / denom  # uv1960-v
    if us < 0.14 or us > 0.32 or vs < 0.25 or vs > 0.38:  # us,vs points are not white area.
        return -1
    dm = 0.0
    for i in range(31):
        di = (vs - uvt[i].v) - uvt[i].t * (us - uvt[i].u)
        if i > 0 and ((di < 0.0 and dm >= 0.0) or (di >= 0.0 and dm < 0.0)):
            break  # found lines bounding (us, vs) : i-1 and i
        dm = di
    if i == 31:
        return -1  # bad XYZ input, color temp would be less than minimum of 1666.7 degrees, or too far towards blue
    di = di / sqrt(1.0 + uvt[i].t * uvt[i].t)
    dm = dm / sqrt(1.0 + uvt[i - 1].t * uvt[i - 1].t)
    p = dm / (dm - di)  # p = interpolation parameter, 0.0 : i-1, 1.0 : i
    p = 1.0 / (LERP(rt[i - 1], rt[i], p))
    temp = p
    return temp
'''


'''class CCTColor:
    def __init__(self, center=None, env=None):
        if env is not None:
            self.MTB = env
            self.MTB.version["_mtbcct"] = VERSION
        self._cct_offset = 0 if center is None else center-6600

        self._cct = np.array([
            1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,
            5200,5400,5600,5800,6000,6200,6400,6600,6800,7000,7200,7400,7600,7800,8000,8200,8400,8600,8800,9000,9200,
            9400,9600,9800,10000,10200,10400,10600,10800,11000,11200,11400,11600,11800,12000,12200,12400,12600,12800,
            13000,13200,13400,13600,13800,14000,14200,14400,14600,14800,15000,15200,15400,15600,15800,16000,16200,16400,
            16600,16800,17000,17200,17400,17600,17800,18000,18200,18400,18600,18800,19000,19200,19400,19600,19800,20000])
        self._red = np.array([
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,249,242,236,231,227,224,221,218,215,213,211,209,207,206,204,203,201,200,199,197,196,195,194,193,192,191,191,
            190,189,188,187,187,186,185,185,184,183,183,182,182,181,181,180,180,179,179,178,178,177,177,176,176,175,175,175,
            174,174,173,173,173,172,172,172,171,171,171,170])
        self._green = np.array([
            67,86,101,114,126,136,146,155,162,170,177,183,189,195,200,205,210,215,219,223,228,231,235,239,242,246,249,252,255,
            246,242,238,236,233,231,229,228,226,225,224,222,221,220,219,218,218,217,216,215,215,214,213,213,212,212,211,210,210,
            209,209,209,208,208,207,207,206,206,206,205,205,205,204,204,204,203,203,203,202,202,202,202,201,201,201,200,200,200,
            200,199,199,199,199,199,198,198,198])
        self._blue = np.array([
            0,0,0,0,0,13,39,60,79,95,109,123,135,146,156,166,175,183,191,198,205,212,219,225,231,236,242,247,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,
            255,255,255,255,255])
        self._cct += self._cct_offset
        # self._rfunc = interp1d(self._cct, self._red)  # Scipy를 사용하는 경우
        # self._gfunc = interp1d(self._cct, self._green)  # Scipy를 사용하는 경우
        # self._bfunc = interp1d(self._cct, self._blue)  # Scipy를 사용하는 경우
        self._rfunc = np.interp(self._cct, self._cct, self._red)  # Numpy를 사용하는 경우
        self._gfunc = np.interp(self._cct, self._cct, self._green)  # Numpy를 사용하는 경우
        self._bfunc = np.interp(self._cct, self._cct, self._blue)  # Numpy를 사용하는 경우

    def cct_to_rgb(self, cct):
        _cct_shape = cct.shape
        cct_flat = cct.flatten()
        cct_flat = np.where(cct_flat < self._cct[0], self._cct[0], cct_flat)
        cct_flat = np.where(cct_flat > self._cct[-1], self._cct[-1], cct_flat)
        # r_flat = self._rfunc(cct_flat)  # Scipy를 사용하는 경우
        # g_flat = self._gfunc(cct_flat)  # Scipy를 사용하는 경우
        # b_flat = self._bfunc(cct_flat)  # Scipy를 사용하는 경우
        r_flat = np.interp(cct_flat, self._cct, self._rfunc)  # Scipy를 사용하는 경우
        g_flat = np.interp(cct_flat, self._cct, self._gfunc)  # Scipy를 사용하는 경우
        b_flat = np.interp(cct_flat, self._cct, self._bfunc)  # Scipy를 사용하는 경우
        return r_flat.reshape(_cct_shape), g_flat.reshape(_cct_shape), b_flat.reshape(_cct_shape)
'''



"""
# Example usage
xy_values = [
    [0.44757,0.40745],[0.34842,0.35161],[0.31006,0.31616],[0.34567,0.3585],[0.33242,0.34743],
    [0.31271,0.32902],[0.29902,0.31485],[0.28315,0.29711],[0.33333,0.33333],[0.3131,0.33727],
    [0.34609,0.35986],[0.38052,0.37713],[0.43695,0.40441],[0.37208,0.37529],[0.4091,0.3943],
    [0.44018,0.40329],[0.31379,0.34531],[0.3779,0.38835],[0.31292,0.32933],[0.34588,0.35875],
    [0.37417,0.37281],[0.456,0.4078],[0.4357,0.4012],[0.3756,0.3723],[0.3422,0.3502],
    [0.3118,0.3236],[0.4474,0.4066],[0.4557,0.4211],[0.456,0.4548],[0.3781,0.3775],
    [0.281, 0.288],
]
color_temp = []
for xy in xy_values:
    color_temp.append(xy_to_CCT(xy))

for i in range(len(xy_values)):
    print(f"{color_temp[i]}")
    # print(f"{xy_values[i]}: CCT={color_temp[i]}")
"""