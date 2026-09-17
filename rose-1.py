"""
粒子玫瑰花束 · 线条勾勒 + 呼吸散开
本次重点改进（玫瑰造型重写）：
- 花瓣宽度采用「根部窄 → 中上部最宽 → 顶部圆钝收口」的包络，不再尖刺
- 花瓣层数加多，层层错位重叠，形成玫瑰花典型的杯状层叠感
- 每片花瓣沿长度方向带内卷弧线（spiral），不是直挺挺的扇形
- 花心是多圈紧密螺旋，外圈逐渐张开
- 外层花瓣向下翻卷，整体形成"花杯"轮廓
- 每片花瓣都有闭合轮廓线，线条和粒子严格同源
"""

import taichi as ti
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import math
import random
import os
import glob

ti.init(arch=ti.gpu)

# ============================================================
# 全局参数
# ============================================================
W, H = 1080, 1080
TOTAL_FRAMES = 400
FPS = 30
N = 45000

ROTATE_SPEED = 0.5
BREATH_CYCLE = 5.0

PARTICLE_RADIUS_BASE = 1.6
FADE_IN_FRAMES = 20

BLOOM_THRESHOLD = 130
BLOOM_SIGMA = 16
BLOOM_INTENSITY = 0.95

N_FLOATING_LIGHTS = 260
OUTPUT_DIR = "./bouquet_output"

LINE_W_PETAL = 1.25
LINE_W_HEART = 1.40
LINE_W_LEAF  = 1.10
LINE_W_VEIN  = 0.90
LINE_W_STEM  = 1.80

ROSE_REDS = [
    (206, 32, 48),
    (232, 56, 66),
    (168, 22, 38),
    (246, 92, 100),
    (222, 46, 60),
    (188, 28, 52),
    (250, 128, 132),
]

COL_LEAF_DARK  = (0.07, 0.20, 0.10)
COL_LEAF_MID   = (0.13, 0.38, 0.18)
COL_LEAF_LIGHT = (0.40, 0.70, 0.34)


# ============================================================
# Taichi 字段（粒子）
# ============================================================
anchor    = ti.Vector.field(3, ti.f32, N)
line_pos  = ti.Vector.field(2, ti.f32, N)
scatter_dir = ti.Vector.field(2, ti.f32, N)
scatter_dist = ti.field(ti.f32, N)
local_z   = ti.field(ti.f32, N)
color     = ti.Vector.field(3, ti.f32, N)
size      = ti.field(ti.f32, N)


def _lerp3(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


# ============================================================
# 线条数据结构
# ============================================================
anchor_list  = []
poly_anchor  = []
poly_pts     = []
poly_z       = []
poly_col     = []
poly_wid     = []

stem_pts = []
stem_col = []
stem_wid = []

ANCHOR_NP = None
PIDX      = None
POLYS_BY_ANCHOR = None


# ============================================================
# 玫瑰 —— 统一的花瓣几何
# ============================================================
# 每层：(花瓣数, 起半径, 末半径, 角度展开, 扫掠角, z 高度, 亮度)
ROSE_LAYERS = [
    ( 4, 0.04, 0.11, 1.75, 0.85,  0.28, 0.36),
    ( 5, 0.06, 0.18, 1.55, 0.68,  0.25, 0.46),
    ( 6, 0.09, 0.27, 1.42, 0.55,  0.21, 0.56),
    ( 7, 0.13, 0.37, 1.32, 0.45,  0.16, 0.66),
    ( 8, 0.18, 0.48, 1.24, 0.38,  0.11, 0.75),
    ( 9, 0.24, 0.60, 1.17, 0.32,  0.05, 0.84),
    (10, 0.30, 0.72, 1.12, 0.27, -0.01, 0.91),
    (11, 0.36, 0.84, 1.09, 0.23, -0.08, 0.96),
    (12, 0.42, 0.96, 1.06, 0.20, -0.16, 1.00),
    (13, 0.48, 1.08, 1.04, 0.18, -0.24, 1.00),
]


def _petal_width_env(u):
    """
    花瓣宽度包络：根部窄 → 中上部最宽 → 顶部圆钝收口。
    峰值大致在 u≈0.55~0.60。
    """
    if u <= 0.0 or u >= 1.0:
        return 0.0
    return math.sin(math.pi * (u ** 1.2)) ** 0.55


def _petal_centerline(radius, a0, r0, r1, sw, u):
    """
    返回花瓣中心线在参数 u 处的位置与切向。
    半径增长不是线性，而是 u^0.85：根部紧凑，外端舒展。
    """
    r_u = r0 + (r1 - r0) * (u ** 0.85)
    theta_u = a0 + sw * (u ** 0.95)      # 顶部卷曲更多
    x_c = r_u * math.cos(theta_u)
    y_c = r_u * math.sin(theta_u)
    return x_c, y_c, theta_u, r_u


def gen_rose(radius, col_main, rng):
    pts, dirs, dists, lz, cols, szs = [], [], [], [], [], []
    cn = np.array(col_main, dtype=np.float32) / 255.0

    def add(x, y, z, c, s):
        pts.append((x, y))
        r = math.hypot(x, y)
        if r > 0.5:
            dirs.append((x / r, y / r))
        else:
            dirs.append((0.0, 0.0))
        dists.append(0.15 * r + 0.02 * radius)
        lz.append(z)
        cols.append(c)
        szs.append(s)

    def shade(bright, u, edge=False, inner=False):
        # 中心暗 → 外圈亮
        v = bright * (0.42 + 0.62 * u)
        if inner:
            v *= 0.55
        if edge:
            v *= 0.70
        c = np.clip(cn * v, 0.0, 1.0)
        return float(c[0]), float(c[1]), float(c[2])

    # --------------------------------------------------------
    # 花心：多圈紧密螺旋（就是真实玫瑰中心那种卷曲）
    # --------------------------------------------------------
    n_sp = 90
    for k in range(n_sp):
        u = k / (n_sp - 1)
        r = radius * (0.005 + 0.085 * (u ** 0.75))
        theta = 0.35 + u * 8.5 * math.pi       # 4 圈多
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = radius * (0.30 - 0.08 * u)
        c = shade(0.40, u, inner=True)
        add(x, y, z, c, 0.88)

        # 螺旋的两条暗边，强化"卷"的感觉
        if 0.10 < u < 0.95:
            for side in (-1, 1):
                ee = theta + side * 0.22
                ex = r * 1.06 * math.cos(ee)
                ey = r * 1.06 * math.sin(ee)
                ec = shade(0.40, u, edge=True, inner=True)
                add(ex, ey, z - radius * 0.004, ec, 0.58)

    # --------------------------------------------------------
    # 花瓣层
    # --------------------------------------------------------
    for li, (n_petals, rb, rt, span, sweep, zh, bright) in enumerate(ROSE_LAYERS):
        avg_angle = math.tau / n_petals
        # 关键：每层旋转偏移，让花瓣错位叠压（黄金角近似）
        rot0 = li * 0.40 + (rng.random() - 0.5) * 0.08

        for i in range(n_petals):
            a0 = i * avg_angle + rot0
            r0 = radius * rb * rng.uniform(0.97, 1.03)
            r1 = radius * rt * rng.uniform(0.97, 1.03)
            sw = sweep * rng.uniform(0.90, 1.10)
            z_base = radius * zh

            # 每片花瓣：纵向采样 + 横向 3 条流线（中心、两条边缘）
            n_u = 12
            for k in range(n_u):
                u = k / (n_u - 1)
                x_c, y_c, theta_u, r_u = _petal_centerline(
                    radius, a0, r0, r1, sw, u)

                # 宽度
                w_env = _petal_width_env(u)
                half_angle = avg_angle * span * 0.5 * w_env

                # 杯状拱起 + 外层向下翻卷
                arch = math.sin(math.pi * u) ** 0.85
                flip_t = max(0.0, u - 0.55) / 0.45
                flip = (flip_t ** 1.6) * (0.10 + 0.045 * li)

                z_c = z_base + radius * (0.055 * arch - flip)

                # ---- 中心脊线 ----
                c_center = shade(bright, u)
                add(x_c, y_c, z_c, c_center,
                    0.88 + 0.10 * math.sin(math.pi * u))

                # ---- 左右边缘 + 内侧过渡 ----
                if 0.03 < u < 0.97:
                    for side in (-1, 1):
                        theta_e = theta_u + side * half_angle
                        re = r_u * (1.0 + 0.03 * math.sin(math.pi * u))
                        xe = re * math.cos(theta_e)
                        ye = re * math.sin(theta_e)
                        # 边缘下沉，形成杯底
                        edge_drop = radius * (0.018 + 0.035 * math.sin(math.pi * u))
                        ze = z_c - edge_drop
                        c_edge = shade(bright, u, edge=True)
                        add(xe, ye, ze, c_edge, 0.72)

                        theta_m = theta_u + side * half_angle * 0.58
                        xm = r_u * math.cos(theta_m)
                        ym = r_u * math.sin(theta_m)
                        zm = z_c - edge_drop * 0.50
                        cm = shade(bright, u)
                        cm = tuple(v * 0.86 for v in cm)
                        add(xm, ym, zm, cm, 0.72)

    return pts, dirs, dists, lz, cols, szs


# ============================================================
# 玫瑰 —— 线条轮廓（与粒子同一套几何）
# ============================================================
def gen_rose_outline(radius, col_main, rng):
    out = []
    cn = np.array(col_main, dtype=np.float32) / 255.0

    def ocol(bright):
        v = 0.52 + 0.55 * bright
        c = np.clip(cn * v, 0.0, 1.0)
        return (float(c[0]), float(c[1]), float(c[2]))

    # ---- 花心螺旋线 ----
    spiral = []
    n_sp = 70
    for k in range(n_sp):
        u = k / (n_sp - 1)
        r = radius * (0.005 + 0.085 * (u ** 0.75))
        theta = 0.35 + u * 8.5 * math.pi
        spiral.append((r * math.cos(theta), r * math.sin(theta)))
    out.append((spiral, radius * 0.26, ocol(0.85), LINE_W_HEART))

    # ---- 花瓣闭合轮廓 ----
    for li, (n_petals, rb, rt, span, sweep, zh, bright) in enumerate(ROSE_LAYERS):
        avg_angle = math.tau / n_petals
        rot0 = li * 0.40 + (rng.random() - 0.5) * 0.08

        for i in range(n_petals):
            a0 = i * avg_angle + rot0
            r0 = radius * rb * rng.uniform(0.97, 1.03)
            r1 = radius * rt * rng.uniform(0.97, 1.03)
            sw = sweep * rng.uniform(0.90, 1.10)

            left, right = [], []
            n_out = 12
            for k in range(n_out):
                u = k / (n_out - 1)
                x_c, y_c, theta_u, r_u = _petal_centerline(
                    radius, a0, r0, r1, sw, u)
                w_env = _petal_width_env(u)
                half_angle = avg_angle * span * 0.5 * w_env

                for side, bucket in ((-1, left), (1, right)):
                    theta_e = theta_u + side * half_angle
                    re = r_u * (1.0 + 0.03 * math.sin(math.pi * u))
                    bucket.append((re * math.cos(theta_e),
                                   re * math.sin(theta_e)))

            loop = left + right[::-1]
            z_mid = radius * zh + radius * 0.045
            out.append((loop, z_mid, ocol(bright), LINE_W_PETAL))

    return out


# ============================================================
# 叶子（保持原逻辑）
# ============================================================
def gen_leaflet(length, rng):
    pts, dirs, dists, lz, cols, szs = [], [], [], [], [], []
    half_w = length * 0.20

    def add(x, y, z, c, s):
        pts.append((x, y))
        r = math.sqrt(x * x + y * y) + 1e-6
        dirs.append((x / r, y / r))
        dists.append(0.35 * r + 0.03 * length)
        lz.append(z)
        cols.append(c)
        szs.append(s)

    n_pts = 20
    for k in range(n_pts):
        u = k / (n_pts - 1)
        x = u * length
        w = half_w * (math.sin(u * math.pi) ** 0.65)
        w *= (1.0 + 0.12 * math.sin(u * 20.0))
        z = math.sin(u * math.pi) * length * 0.04
        c = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.25 + 0.55 * rng.random())
        add(x, w, z, c, 0.78)
        c2 = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.25 + 0.55 * rng.random())
        add(x, -w, z, c2, 0.78)

    for k in range(n_pts):
        u = k / (n_pts - 1)
        x = u * length
        z = math.sin(u * math.pi) * length * 0.05
        c = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.65)
        add(x, 0, z, c, 0.68)

    for k in range(3, n_pts - 3, 3):
        u = k / (n_pts - 1)
        x0 = u * length
        w0 = half_w * (math.sin(u * math.pi) ** 0.65) * 0.80
        z0 = math.sin(u * math.pi) * length * 0.05
        for side in (-1, 1):
            for tt in (0.45, 0.85):
                px = x0 + length * 0.13 * tt
                py = side * w0 * tt
                c = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.45)
                add(px, py, z0, c, 0.62)

    return pts, dirs, dists, lz, cols, szs


def gen_compound_leaf(length, rng):
    pts, dirs, dists, lz, cols, szs = [], [], [], [], [], []

    layout = [
        ( 0.00, 0.00, 1.00),
        ( 0.80, 0.18, 0.70),
        (-0.80, 0.18, 0.70),
    ]
    if length > 75:
        layout += [
            ( 1.40, 0.40, 0.52),
            (-1.40, 0.40, 0.52),
        ]

    for ang, bx, sc in layout:
        L = length * sc
        lp, ld, ldist, llz, lc, ls = gen_leaflet(L, rng)
        ca, sa = math.cos(ang), math.sin(ang)
        for (x, y), (dx, dy), dd, zz, c, s in zip(lp, ld, ldist, llz, lc, ls):
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rdx = dx * ca - dy * sa
            rdy = dx * sa + dy * ca
            pts.append((bx * length + rx, ry))
            dirs.append((rdx, rdy))
            dists.append(dd)
            lz.append(zz)
            cols.append(c)
            szs.append(s)

    col_pet = _lerp3(COL_LEAF_DARK, COL_LEAF_MID, 0.35)
    for k in range(10):
        t = k / 9.0
        x = -t * length * 0.55
        pts.append((x, 0))
        dirs.append((0, 0))
        dists.append(0.03 * length)
        lz.append(0)
        cols.append(col_pet)
        szs.append(0.72)

    return pts, dirs, dists, lz, cols, szs


def gen_leaf_outline(length, angle, rng):
    out = []

    layout = [
        ( 0.00, 0.00, 1.00),
        ( 0.80, 0.18, 0.70),
        (-0.80, 0.18, 0.70),
    ]
    if length > 75:
        layout += [
            ( 1.40, 0.40, 0.52),
            (-1.40, 0.40, 0.52),
        ]

    ca0, sa0 = math.cos(angle), math.sin(angle)

    def place(x, y):
        return (x * ca0 - y * sa0, x * sa0 + y * ca0)

    edge_col = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.62)
    vein_col = _lerp3(COL_LEAF_DARK, COL_LEAF_LIGHT, 0.42)

    for ang, bx, sc in layout:
        L = length * sc
        ca, sa = math.cos(ang), math.sin(ang)

        def local(x, y):
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            return place(bx * length + rx, ry)

        n = 11
        half_w = L * 0.20
        up, lo = [], []
        for k in range(n):
            u = k / (n - 1)
            x = u * L
            w = half_w * (math.sin(u * math.pi) ** 0.65)
            w *= (1.0 + 0.12 * math.sin(u * 20.0))
            up.append(local(x, w))
            lo.append(local(x, -w))

        out.append((up + lo[::-1], length * 0.05, edge_col, LINE_W_LEAF))

        vein = [local(k / (n - 1) * L, 0.0) for k in range(n)]
        out.append((vein, length * 0.06, vein_col, LINE_W_VEIN))

    pet = [place(-k / 9.0 * length * 0.55, 0.0) for k in range(10)]
    out.append((pet, 0.0, _lerp3(COL_LEAF_DARK, COL_LEAF_MID, 0.40), LINE_W_LEAF))

    return out


# ============================================================
# 场景构建
# ============================================================
billboard_data = []
DEPTH_NP = None


def build_scene():
    global billboard_data, DEPTH_NP, ANCHOR_NP, PIDX, POLYS_BY_ANCHOR

    billboard_data = []
    anchor_list.clear()
    poly_anchor.clear(); poly_pts.clear(); poly_z.clear()
    poly_col.clear(); poly_wid.clear()
    stem_pts.clear(); stem_col.clear(); stem_wid.clear()

    rng = random.Random(20240517)

    CX, CY, CZ = 500.0, 500.0, 0.0
    R_horiz = 250.0
    R_vert = 200.0

    # ---------------- 花朵 ----------------
    flower_layers = [
        (-0.92,  3, 0.30),
        (-0.70,  4, 0.58),
        (-0.45,  5, 0.80),
        (-0.18,  6, 0.96),
        ( 0.08,  6, 1.00),
    ]

    flower_positions = []

    for li, (h, n, hr) in enumerate(flower_layers):
        y_base = CY + h * R_vert
        for i in range(n):
            theta = (i / n) * math.pi * 2 + li * 0.50
            r_th = R_horiz * hr
            x = CX + r_th * math.cos(theta) + rng.uniform(-10, 10)
            y = y_base + rng.uniform(-12, 12)
            z = CZ + r_th * math.sin(theta) * 0.75 + rng.uniform(-10, 10)

            depth_t01 = ((z - CZ) / R_horiz + 1) * 0.5
            fr = 62.0 + depth_t01 * 28.0

            col = ROSE_REDS[(li * 5 + i) % len(ROSE_REDS)]

            aidx = len(anchor_list)
            anchor_list.append((x, y, z))
            flower_positions.append((x, y, z))

            # 粒子
            p_, d_, dist_, lz_, c_, s_ = gen_rose(fr, col, rng)
            for (lx, ly), (dx, dy), dd, zz, c, s in zip(p_, d_, dist_, lz_, c_, s_):
                billboard_data.append((aidx, (x, y, z), (lx, ly), (dx, dy), dd, zz, c, s))

            # 线条轮廓
            for (pts, zz, cc, ww) in gen_rose_outline(fr, col, rng):
                poly_anchor.append(aidx)
                poly_pts.append(pts)
                poly_z.append(zz)
                poly_col.append(cc)
                poly_wid.append(ww)

    # ---------------- 叶子 ----------------
    leaf_layers = [
        (0.16,  8, 1.00),
        (0.44, 10, 0.92),
        (0.74, 12, 0.70),
    ]

    for li, (h, n, hr) in enumerate(leaf_layers):
        y_base = CY + h * R_vert
        for i in range(n):
            theta = (i / n) * math.pi * 2 + li * 0.37
            r_th = R_horiz * hr
            x = CX + r_th * math.cos(theta) + rng.uniform(-5, 5)
            y = y_base + rng.uniform(-6, 6)
            z = CZ + r_th * math.sin(theta) * 0.75 + rng.uniform(-5, 5)

            angle = theta + rng.uniform(-0.40, 0.40)
            leaf_len = 95.0 + rng.uniform(-15, 20)

            aidx = len(anchor_list)
            anchor_list.append((x, y, z))

            p_, d_, dist_, lz_, c_, s_ = gen_compound_leaf(leaf_len, rng)
            ca, sa = math.cos(angle), math.sin(angle)
            for (lx, ly), (dx, dy), dd, zz, c, s in zip(p_, d_, dist_, lz_, c_, s_):
                rx = lx * ca - ly * sa
                ry = lx * sa + ly * ca
                rdx = dx * ca - dy * sa
                rdy = dx * sa + dy * ca
                billboard_data.append((aidx, (x, y, z), (rx, ry), (rdx, rdy), dd, zz, c, s))

            for (pts, zz, cc, ww) in gen_leaf_outline(leaf_len, angle, rng):
                poly_anchor.append(aidx)
                poly_pts.append(pts)
                poly_z.append(zz)
                poly_col.append(cc)
                poly_wid.append(ww)

    # ---------------- 花枝 ----------------
    base = np.array([CX, CY - 245.0, CZ], dtype=np.float32)
    for (fx, fy, fz) in flower_positions:
        p0 = np.array([fx, fy, fz], dtype=np.float32)
        ctrl = (p0 + base) * 0.5 + np.array([0.0, 0.0, 45.0], dtype=np.float32)
        pts = []
        n_seg = 12
        for k in range(n_seg):
            u = k / (n_seg - 1)
            pt = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * ctrl + u ** 2 * base
            pts.append((float(pt[0]), float(pt[1]), float(pt[2])))
        stem_pts.append(pts)
        stem_col.append((0.16, 0.42, 0.20))
        stem_wid.append(LINE_W_STEM)

    print(f"原始粒子数：{len(billboard_data)}，折线数：{len(poly_anchor)}")

    if len(billboard_data) < N:
        original = len(billboard_data)
        while len(billboard_data) < N:
            k = random.randint(0, original - 1)
            (ai, anc, lp, sd, dd, zz, c, s) = billboard_data[k]
            new_lp = (lp[0] + random.uniform(-0.8, 0.8),
                      lp[1] + random.uniform(-0.8, 0.8))
            billboard_data.append((ai, anc, new_lp, sd, dd, zz, c, s * 0.95))

    if len(billboard_data) > N:
        idx = np.random.choice(len(billboard_data), N, replace=False)
        billboard_data = [billboard_data[i] for i in idx]

    anchor_np = np.zeros((N, 3), dtype=np.float32)
    line_np   = np.zeros((N, 2), dtype=np.float32)
    sdir_np   = np.zeros((N, 2), dtype=np.float32)
    sdist_np  = np.zeros(N, dtype=np.float32)
    lz_np     = np.zeros(N, dtype=np.float32)
    color_np  = np.zeros((N, 3), dtype=np.float32)
    size_np   = np.zeros(N, dtype=np.float32)
    pidx_np   = np.zeros(N, dtype=np.int32)

    for i, (ai, anc, lp, sd, dd, zz, c, s) in enumerate(billboard_data):
        pidx_np[i] = ai
        anchor_np[i] = anc
        line_np[i] = lp
        sdir_np[i] = sd
        sdist_np[i] = dd
        lz_np[i] = zz
        color_np[i] = c
        size_np[i] = s

    anchor.from_numpy(anchor_np)
    line_pos.from_numpy(line_np)
    scatter_dir.from_numpy(sdir_np)
    scatter_dist.from_numpy(sdist_np)
    local_z.from_numpy(lz_np)
    color.from_numpy(color_np)
    size.from_numpy(size_np)

    DEPTH_NP = lz_np

    n_anchors = len(anchor_list)
    ANCHOR_NP = np.array(anchor_list, dtype=np.float32)
    PIDX = pidx_np

    POLYS_BY_ANCHOR = [[] for _ in range(n_anchors)]
    for i, a in enumerate(poly_anchor):
        POLYS_BY_ANCHOR[a].append(i)

    print(f"场景完成：{N} 粒子 / {n_anchors} 锚点 / {len(poly_anchor)} 条折线")


# ============================================================
# 背景
# ============================================================
bg_image = None


def build_background():
    global bg_image
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W / 2.0, H / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    maxd = math.sqrt(cx * cx + cy * cy)
    t = np.clip(dist / maxd, 0, 1)

    r = 55 * (1 - t) ** 2.6 + 6
    g = 28 * (1 - t) ** 2.6 + 4
    b = 80 * (1 - t) ** 2.6 + 16

    hx, hy = W / 2.0, H * 0.45
    halo_dist = np.sqrt((xx - hx) ** 2 + (yy - hy) ** 2)
    halo_t = np.clip(halo_dist / (max(W, H) * 0.55), 0, 1)
    halo = (1 - halo_t) ** 2.8
    r += 60 * halo
    g += 25 * halo
    b += 22 * halo

    bg = np.stack([r, g, b], axis=2)
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bg_image = Image.fromarray(bg, mode="RGB")

    star_draw = ImageDraw.Draw(bg_image)
    for _ in range(400):
        sx = random.randint(0, W - 1)
        sy = random.randint(0, H - 1)
        dx = sx - W / 2
        dy = sy - H / 2
        if math.sqrt(dx * dx + dy * dy) < 260:
            continue
        b_val = random.randint(60, 180)
        star_draw.point((sx, sy), fill=(int(b_val * 0.8), int(b_val * 0.85), b_val))
        if random.random() < 0.15:
            star_draw.point((sx + 1, sy),
                            fill=(int(b_val * 0.5), int(b_val * 0.55), int(b_val * 0.7)))


# ============================================================
# 飘散光点
# ============================================================
class FloatingLight:
    def __init__(self):
        self.reset(initial=True)

    def reset(self, initial=False):
        self.x = random.uniform(100, 900)
        self.y = random.uniform(100, 900) if initial else 1000
        self.z = random.uniform(-200, 200)
        self.vx = random.uniform(-0.35, 0.35)
        self.vy = random.uniform(-0.9, -0.3)
        self.vz = random.uniform(-0.18, 0.18)
        self.size = random.uniform(1.5, 4.5)
        self.color = random.choice([
            (255, 180, 100), (255, 120, 130),
            (255, 200, 140), (255, 90, 110),
        ])
        self.phase = random.uniform(0, 6.28)

    def update(self, t):
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        if self.y < 80:
            self.reset(initial=False)


# ============================================================
# 渲染
# ============================================================
def render_frame(frame_idx, t, floating_lights):
    img = bg_image.copy()
    draw = ImageDraw.Draw(img)

    ay_rot = t * ROTATE_SPEED
    cos_y, sin_y = math.cos(ay_rot), math.sin(ay_rot)

    focal = 900.0
    cx_screen = W * 0.5
    cy_screen = H * 0.5

    osc = 0.5 - 0.5 * math.cos(t * 2 * math.pi / BREATH_CYCLE)

    global_alpha = min(1.0, frame_idx / float(FADE_IN_FRAMES))
    if global_alpha <= 0.0:
        global_alpha = 0.0

    # 锚点投影
    aax = ANCHOR_NP[:, 0] - 500.0
    aay = ANCHOR_NP[:, 1] - 500.0
    aaz = ANCHOR_NP[:, 2]
    arx = aax * cos_y + aaz * sin_y
    arz = -aax * sin_y + aaz * cos_y
    ary = aay
    ascale = np.clip(focal / (focal + arz), 0.35, 2.8)
    asx = cx_screen + arx * ascale
    asy = cy_screen + ary * ascale

    # 粒子投影
    anchor_np = anchor.to_numpy()
    line_np = line_pos.to_numpy()
    sdir_np = scatter_dir.to_numpy()
    sdist_np = scatter_dist.to_numpy()
    lz_np = local_z.to_numpy()
    color_np = color.to_numpy()
    size_np = size.to_numpy()

    ax = anchor_np[:, 0] - 500.0
    ay = anchor_np[:, 1] - 500.0
    az = anchor_np[:, 2]
    rx = ax * cos_y + az * sin_y
    rz = -ax * sin_y + az * cos_y
    ry = ay

    scale_anchor = np.clip(focal / (focal + rz), 0.35, 2.8)
    sx_anchor = cx_screen + rx * scale_anchor
    sy_anchor = cy_screen + ry * scale_anchor

    lx = line_np[:, 0] + sdir_np[:, 0] * sdist_np * osc
    ly = line_np[:, 1] + sdir_np[:, 1] * sdist_np * osc

    px_all = sx_anchor + lx * scale_anchor
    py_all = sy_anchor + ly * scale_anchor
    depth_all = rz + lz_np * scale_anchor + osc * sdist_np * 0.15

    size_scale = 0.85 + 0.30 * osc
    bright_scale = 1.0 - 0.12 * osc

    # 花枝
    if global_alpha > 0.01:
        stem_draw = []
        for pts3, col, wid in zip(stem_pts, stem_col, stem_wid):
            proj = []
            zsum = 0.0
            for (wx, wy, wz) in pts3:
                X = wx - 500.0
                Y = wy - 500.0
                Z = wz
                RX = X * cos_y + Z * sin_y
                RZ = -X * sin_y + Z * cos_y
                RY = Y
                sc = focal / (focal + RZ)
                sc = max(0.35, min(2.8, sc))
                proj.append((cx_screen + RX * sc, cy_screen + RY * sc))
                zsum += RZ
            zavg = zsum / max(1, len(pts3))
            stem_draw.append((zavg, proj, col, wid))

        stem_draw.sort(key=lambda v: -v[0])
        for zavg, proj, col, wid in stem_draw:
            df = focal / (focal + zavg)
            df = max(0.45, min(1.15, df))
            cc = (int(col[0] * 255 * global_alpha * df),
                  int(col[1] * 255 * global_alpha * df),
                  int(col[2] * 255 * global_alpha * df))
            cc = (max(0, min(255, cc[0])),
                  max(0, min(255, cc[1])),
                  max(0, min(255, cc[2])))
            try:
                draw.line(proj, fill=cc, width=max(1, int(round(wid))), joint="curve")
            except Exception:
                draw.line(proj, fill=cc, width=max(1, int(round(wid))))

    # 排序
    n_anchors = len(ANCHOR_NP)
    rank = np.empty(n_anchors, dtype=np.int32)
    rank[np.argsort(-arz)] = np.arange(n_anchors, dtype=np.int32)

    order = np.lexsort((-depth_all, rank[PIDX]))

    def draw_anchor_lines(a):
        s = float(ascale[a])
        if s < 0.01:
            return
        ox = float(asx[a])
        oy = float(asy[a])
        bz = float(arz[a])
        for pi in POLYS_BY_ANCHOR[a]:
            pts = poly_pts[pi]
            proj = [(ox + p[0] * s, oy + p[1] * s) for p in pts]
            if len(proj) < 2:
                continue
            zz = bz + poly_z[pi] * s
            df = focal / (focal + zz)
            if df < 0.45:
                df = 0.45
            elif df > 1.15:
                df = 1.15
            c = poly_col[pi]
            cr = int(c[0] * 255 * global_alpha * df * bright_scale)
            cg = int(c[1] * 255 * global_alpha * df * bright_scale)
            cb = int(c[2] * 255 * global_alpha * df * bright_scale)
            cr = max(0, min(255, cr))
            cg = max(0, min(255, cg))
            cb = max(0, min(255, cb))
            w = int(round(poly_wid[pi] * s))
            if w < 1:
                w = 1
            elif w > 4:
                w = 4
            try:
                draw.line(proj, fill=(cr, cg, cb), width=w, joint="curve")
            except Exception:
                draw.line(proj, fill=(cr, cg, cb), width=w)

    prev_a = -1
    for idx in order:
        a = int(PIDX[idx])
        if a != prev_a:
            if prev_a >= 0 and global_alpha > 0.01:
                draw_anchor_lines(prev_a)
            prev_a = a

        px = px_all[idx]
        py = py_all[idx]
        if px < -30 or px > W + 30 or py < -30 or py > H + 30:
            continue

        z_val = depth_all[idx]
        depth_factor = focal / (focal + z_val)
        depth_factor = max(0.45, min(1.15, depth_factor))

        r = PARTICLE_RADIUS_BASE * size_np[idx] * scale_anchor[idx] * size_scale
        r = max(0.4, min(6.0, r))

        cr = int(color_np[idx, 0] * 255 * global_alpha * depth_factor * bright_scale)
        cg = int(color_np[idx, 1] * 255 * global_alpha * depth_factor * bright_scale)
        cb = int(color_np[idx, 2] * 255 * global_alpha * depth_factor * bright_scale)
        cr = max(0, min(255, cr))
        cg = max(0, min(255, cg))
        cb = max(0, min(255, cb))

        if r < 0.8:
            draw.point((px, py), fill=(cr, cg, cb))
        else:
            draw.ellipse([px - r, py - r, px + r, py + r], fill=(cr, cg, cb))

    if prev_a >= 0 and global_alpha > 0.01:
        draw_anchor_lines(prev_a)

    # 飘散光点
    light_render = []
    for L in floating_lights:
        L.update(t)
        lax = L.x - 500.0
        lay = L.y - 500.0
        laz = L.z
        lrx = lax * cos_y + laz * sin_y
        lrz = -lax * sin_y + laz * cos_y
        lry = lay
        sc = focal / (focal + lrz)
        sc = max(0.35, min(2.6, sc))
        px = cx_screen + lrx * sc
        py = cy_screen + lry * sc
        if px < -30 or px > W + 30 or py < -30 or py > H + 30:
            continue
        light_render.append((lrz, px, py, L))

    light_render.sort(key=lambda x: -x[0])

    for z_val, px, py, L in light_render:
        depth_factor = focal / (focal + z_val)
        depth_factor = max(0.45, min(1.2, depth_factor))
        tw = 0.5 + 0.5 * math.sin(t * 2.2 + L.phase)
        r = L.size * depth_factor * (0.7 + 0.4 * tw)
        cr = int(L.color[0] * global_alpha * depth_factor * (0.7 + 0.4 * tw))
        cg = int(L.color[1] * global_alpha * depth_factor * (0.7 + 0.4 * tw))
        cb = int(L.color[2] * global_alpha * depth_factor * (0.7 + 0.4 * tw))
        cr = max(0, min(255, cr))
        cg = max(0, min(255, cg))
        cb = max(0, min(255, cb))

        draw.ellipse([px - r * 2.5, py - r * 2.5, px + r * 2.5, py + r * 2.5],
                     fill=(cr // 6, cg // 6, cb // 6))
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(cr, cg, cb))

    return np.array(img)


def apply_bloom(img_np):
    img_f = img_np.astype(np.float32)
    brightness = np.max(img_f, axis=2, keepdims=True)
    mask = (brightness > BLOOM_THRESHOLD).astype(np.float32)
    bright = (img_f * mask).astype(np.uint8)

    bright_img = Image.fromarray(bright, mode="RGB")
    blurred_img = bright_img.filter(ImageFilter.GaussianBlur(radius=BLOOM_SIGMA))
    blurred = np.array(blurred_img).astype(np.float32)

    result = img_f + blurred * BLOOM_INTENSITY
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("构建背景...")
    build_background()

    print("构建场景...")
    build_scene()

    floating_lights = [FloatingLight() for _ in range(N_FLOATING_LIGHTS)]

    vm = ti.tools.VideoManager(
        output_dir=OUTPUT_DIR,
        framerate=FPS,
        automatic_build=False
    )

    print(f"渲染 {TOTAL_FRAMES} 帧...")
    for frame in range(TOTAL_FRAMES):
        t = frame / FPS
        img_np = render_frame(frame, t, floating_lights)
        img_np = apply_bloom(img_np)
        vm.write_frame(img_np)
        if frame % 20 == 0 or frame == TOTAL_FRAMES - 1:
            print(f"  {frame + 1}/{TOTAL_FRAMES}")

    print("导出视频...")
    try:
        vm.make_video(mp4=True, gif=True)
        print(f"MP4: {vm.get_output_filename('.mp4')}")
        print(f"GIF: {vm.get_output_filename('.gif')}")
    except Exception as e:
        print(f"Taichi 编码失败: {e}")
        frame_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*.png")))
        if frame_files:
            images = [Image.open(f) for f in frame_files]
            gif_path = os.path.join(OUTPUT_DIR, "bouquet.gif")
            images[0].save(
                gif_path,
                save_all=True,
                append_images=images[1:],
                duration=int(1000 / FPS),
                loop=0
            )
            print(f"GIF: {gif_path}")

    print("完成。")


if __name__ == "__main__":
    main()