import cv2
import numpy as np

DUST_COLOUR = np.array([200.0, 160.0, 120.0], np.float32)
MAX_DUST_SPOTS = 600
KINDS = ('dust', 'dirt', 'dark', 'both')


def dust_spots(shape, seed=0, n=MAX_DUST_SPOTS):
    rng = np.random.default_rng(seed)
    h, w = shape[:2]
    diag = np.hypot(h, w)
    spots = []
    for _ in range(n):
        x = rng.uniform(0, w)
        y = rng.beta(2.0, 1.3) * h
        rx = min(0.005 * (1 - rng.uniform()) ** (-1 / 1.4), 0.08) * diag
        ry = rx * rng.uniform(0.5, 1.0)
        spots.append((x, y, rx, ry, rng.uniform(0, 180), rng.uniform(0.3, 0.85)))
    return spots


def dust_alpha(shape, spots):
    h, w = shape[:2]
    log_t = np.zeros((h, w), np.float32)
    for x, y, rx, ry, angle, alpha in spots:
        r = int(np.ceil(max(rx, ry))) + 2
        cx, cy = int(x), int(y)
        patch = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
        cv2.ellipse(patch, (r, r), (max(1, int(rx)), max(1, int(ry))), angle, 0, 360, 1.0, -1, cv2.LINE_AA)
        x0, y0 = cx - r, cy - r
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + 2 * r + 1), min(h, y0 + 2 * r + 1)
        if xa >= xb or ya >= yb:
            continue
        log_t[ya:yb, xa:xb] += patch[ya - y0:yb - y0, xa - x0:xb - x0] * np.log(1 - alpha)
    a = 1 - np.exp(log_t)
    return cv2.GaussianBlur(a, (0, 0), 0.006 * np.hypot(h, w))[..., None]


def add_dust(img, severity, seed=0, alpha=None):
    if severity <= 0:
        return img.copy()
    if alpha is None:
        spots = dust_spots(img.shape, seed)
        alpha = dust_alpha(img.shape, spots[:int(round(severity * len(spots)))])
    h, w = img.shape[:2]
    i = img.astype(np.float32)
    small = cv2.resize(i, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    scatter = cv2.resize(cv2.GaussianBlur(small, (0, 0), 0.005 * np.hypot(h, w)), (w, h))
    out = i * (1 - alpha) + (0.65 * DUST_COLOUR + 0.35 * scatter) * alpha
    film = 0.5 * (1 - np.exp(-severity))
    out = out * (1 - film) + film * (0.6 * scatter + 0.4 * DUST_COLOUR)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_darkness(img, severity, seed=None):
    if severity <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    x = img.astype(np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rr = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    vignette = 1 - 0.55 * severity * np.clip(rr / 1.3, 0, 1) ** 2
    exposure = 1 - 0.85 * severity
    x = x * exposure * vignette[..., None]
    x = rng.poisson(x * 600.0).astype(np.float32) / 600.0
    x = x + rng.normal(0, 0.004 + 0.01 * severity, x.shape).astype(np.float32)
    return np.clip(x * exposure ** -0.6 * 255, 0, 255).astype(np.uint8)


def degrade(img, kind, severity, seed=0, dust_alpha_mask=None):
    out = img
    if kind in ('dust', 'dirt', 'both'):
        out = add_dust(out, severity, seed, dust_alpha_mask)
    if kind in ('dark', 'both'):
        out = add_darkness(out, severity, seed)
    return out

