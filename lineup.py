#!/usr/bin/env python3
"""Line up a comparand TIFF to a reference TIFF for visual diffing.

The two images share content but differ by:
  * a global horizontal stretch plus a per-line horizontal shift (geometry), and
  * a global value shift+stretch that may crush blacks/whites (value).

There are no vertical changes; every row is solved independently. The comparand
is resampled onto the reference's grid and written out with the reference's
exact TIFF specification (IFD cloned verbatim).

Usage:  python lineup.py REF COMP OUT
"""

import struct
import sys

import numpy as np
from scipy import ndimage, optimize

# ---------------------------------------------------------------------------
# TIFF I/O (self-contained, little-endian, chunky YCbCr 4:2:2 / PackBits)
# ---------------------------------------------------------------------------

TYPESIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1}

TAG_IMAGEWIDTH = 256
TAG_IMAGELENGTH = 257
TAG_COMPRESSION = 259
TAG_PHOTOMETRIC = 262
TAG_STRIPOFFSETS = 273
TAG_ROWSPERSTRIP = 278
TAG_STRIPBYTECOUNTS = 279
TAG_SUBSAMPLING = 530

COMPRESSION_PACKBITS = 32773
PHOTOMETRIC_YCBCR = 6


def packbits_decode(data, expected):
    """PackBits decompress `data` until `expected` bytes are produced."""
    out = bytearray()
    i = 0
    n = len(data)
    while i < n and len(out) < expected:
        h = data[i]
        i += 1
        if h < 128:
            cnt = h + 1
            out += data[i:i + cnt]
            i += cnt
        elif h > 128:
            cnt = 257 - h
            out += bytes([data[i]]) * cnt
            i += 1
        # h == 128 is a no-op
    return bytes(out)


def packbits_encode(data):
    """PackBits compress `data` (standard literal/run encoding)."""
    out = bytearray()
    n = len(data)
    i = 0
    while i < n:
        # detect a run of >=3 identical bytes
        run = 1
        while i + run < n and data[i + run] == data[i] and run < 128:
            run += 1
        if run >= 3:
            out.append(257 - run)
            out.append(data[i])
            i += run
            continue
        # literal run: collect bytes until a 3+ run starts or 128 reached
        start = i
        i += 1
        while i < n and (i - start) < 128:
            if i + 2 < n and data[i] == data[i + 1] == data[i + 2]:
                break
            i += 1
        lit = data[start:i]
        out.append(len(lit) - 1)
        out += lit
    return bytes(out)


def read_tiff(path):
    """Parse a little-endian TIFF; return dict with tag table and raw planes.

    Returns full-width float64 planes Y, U, V (chroma spline-resampled to full
    width). Raises on non-YCbCr photometric or non-little-endian byte order.
    """
    b = open(path, "rb").read()
    if b[:2] != b"II":
        raise ValueError(f"{path}: only little-endian (II) TIFF supported")
    (ifd_off,) = struct.unpack_from("<I", b, 4)
    (ncnt,) = struct.unpack_from("<H", b, ifd_off)
    order = []
    tags = {}
    for k in range(ncnt):
        off = ifd_off + 2 + k * 12
        tag, typ, cnt = struct.unpack_from("<HHI", b, off)
        vsize = TYPESIZE.get(typ, 1) * cnt
        voff = off + 8 if vsize <= 4 else struct.unpack_from("<I", b, off + 8)[0]
        if typ == 3:
            vals = list(struct.unpack_from("<%dH" % cnt, b, voff))
        elif typ == 4:
            vals = list(struct.unpack_from("<%dI" % cnt, b, voff))
        elif typ == 5:
            vals = [struct.unpack_from("<II", b, voff + 8 * j) for j in range(cnt)]
        elif typ == 2:
            vals = b[voff:voff + cnt]
        else:
            vals = list(b[voff:voff + cnt])
        tags[tag] = [typ, cnt, vals]
        order.append(tag)

    photo = tags[TAG_PHOTOMETRIC][2][0]
    if photo != PHOTOMETRIC_YCBCR:
        raise ValueError(
            f"{path}: Photometric={photo}, expected {PHOTOMETRIC_YCBCR} (YCbCr); "
            "not a YUV image"
        )
    W = tags[TAG_IMAGEWIDTH][2][0]
    H = tags[TAG_IMAGELENGTH][2][0]
    rps = tags[TAG_ROWSPERSTRIP][2][0]
    offs = tags[TAG_STRIPOFFSETS][2]
    counts = tags[TAG_STRIPBYTECOUNTS][2]
    comp = tags[TAG_COMPRESSION][2][0]
    if comp != COMPRESSION_PACKBITS:
        raise ValueError(f"{path}: Compression={comp}, expected PackBits (32773)")

    raw = bytearray()
    for o, c, s in ((offs[i], counts[i], i) for i in range(len(offs))):
        r0 = s * rps
        rows = min(rps, H - r0)
        raw += packbits_decode(b[o:o + c], 2 * W * rows)
    raw = np.frombuffer(bytes(raw[:2 * W * H]), dtype=np.uint8).astype(np.float64)
    raw = raw.reshape(H, 2 * W)  # chunky [Y0,Y1,Cb,Cr] per 2-pixel unit

    sub = tags.get(TAG_SUBSAMPLING)
    sub = list(sub[2]) if sub else [2, 1]

    # Reconstruct planes. Chunky packing for 4:2:2 is groups of 4 bytes
    # [Y0, Y1, Cb, Cr] covering two horizontal luma pixels.
    if sub == [2, 1]:
        groups = raw.reshape(H, W // 2, 4)
        Y = np.empty((H, W), dtype=np.float64)
        Y[:, 0::2] = groups[:, :, 0]
        Y[:, 1::2] = groups[:, :, 1]
        Cb_h = groups[:, :, 2]  # half width
        Cr_h = groups[:, :, 3]
        U = _spline_resample_width(Cb_h, W)
        V = _spline_resample_width(Cr_h, W)
    else:
        # Fallback: treat as full-width interleaved is impossible for other
        # subsamplings without knowing layout; warn and best-effort upsample.
        sys.stderr.write(
            f"warning: {path}: YCbCrSubSampling={sub} is not 4:2:2; "
            "spline-scaling chroma to full width (best effort)\n"
        )
        # Assume luma fills first H*W bytes, chroma follows at subsampled res.
        cw = max(1, W // (sub[0] if sub else 1))
        ch = max(1, H // (sub[1] if len(sub) > 1 else 1))
        flat = raw.reshape(-1)
        Y = flat[:H * W].reshape(H, W)
        rest = flat[H * W:]
        need = ch * cw
        Cb = rest[:need].reshape(ch, cw) if rest.size >= need else np.full((ch, cw), 128.0)
        Cr = rest[need:need * 2].reshape(ch, cw) if rest.size >= need * 2 else np.full((ch, cw), 128.0)
        U = _spline_resample(Cb, (H, W))
        V = _spline_resample(Cr, (H, W))

    return {"b": b, "ifd_off": ifd_off, "order": order, "tags": tags,
            "W": W, "H": H, "sub": sub, "Y": Y, "U": U, "V": V}


def _spline_resample_width(plane, out_w):
    """Cubic-spline resample a (H, w) plane to (H, out_w) along width only."""
    H, w = plane.shape
    if w == out_w:
        return plane.astype(np.float64)
    # map output column j -> input coordinate (centered)
    cols = (np.arange(out_w) + 0.5) * (w / out_w) - 0.5
    cols = np.clip(cols, 0, w - 1)
    rows = np.arange(H)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return ndimage.map_coordinates(plane, [rr, cc], order=3, mode="nearest")


def _spline_resample(plane, out_shape):
    """Cubic-spline resample a 2D plane to out_shape (rows, cols)."""
    H, W = plane.shape
    oh, ow = out_shape
    rows = (np.arange(oh) + 0.5) * (H / oh) - 0.5
    cols = (np.arange(ow) + 0.5) * (W / ow) - 0.5
    rows = np.clip(rows, 0, H - 1)
    cols = np.clip(cols, 0, W - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return ndimage.map_coordinates(plane, [rr, cc], order=3, mode="nearest")


def write_tiff(path, ref, Y, U, V):
    """Write planes Y,U,V using the reference's cloned IFD (4:2:2 PackBits)."""
    tags_src = ref["tags"]
    order = ref["order"]
    W = ref["W"]
    H = ref["H"]
    rps = tags_src[TAG_ROWSPERSTRIP][2][0]

    # Downsample chroma to half width for 4:2:2 packing.
    Cb_h = _spline_resample_width(U, W // 2)
    Cr_h = _spline_resample_width(V, W // 2)

    def q(a):
        return np.clip(np.rint(a), 0, 255).astype(np.uint8)

    Yq = q(Y)
    Cbq = q(Cb_h)
    Crq = q(Cr_h)
    groups = np.empty((H, W // 2, 4), dtype=np.uint8)
    groups[:, :, 0] = Yq[:, 0::2]
    groups[:, :, 1] = Yq[:, 1::2]
    groups[:, :, 2] = Cbq
    groups[:, :, 3] = Crq
    raw = groups.reshape(H, 2 * W)
    # PackBits must be encoded per scanline: ffmpeg (and the TIFF spec) decode
    # one row at a time and reject runs that cross a row boundary ("Copy went
    # out of bounds"). Encode each row separately, then concatenate per strip.
    rows_enc = [packbits_encode(raw[r].tobytes()) for r in range(H)]

    nstrips = (H + rps - 1) // rps
    enc = []
    for s in range(nstrips):
        r0 = s * rps
        r1 = min(r0 + rps, H)
        enc.append(b"".join(rows_enc[r0:r1]))
    counts = [len(e) for e in enc]

    # Clone tag table; override strip offsets/counts.
    tags = {}
    for t in order:
        typ, cnt, vals = tags_src[t]
        v = list(vals) if isinstance(vals, list) else vals
        tags[t] = [typ, cnt, v]
    tags[TAG_STRIPBYTECOUNTS] = [tags_src[TAG_STRIPBYTECOUNTS][0], nstrips, counts]
    tags[TAG_STRIPOFFSETS] = [tags_src[TAG_STRIPOFFSETS][0], nstrips, [0] * nstrips]

    ncnt = len(order)
    header = 8
    ifd_size = 2 + 12 * ncnt + 4

    def build(soffs):
        """Serialize external value blocks + IFD given strip offsets."""
        ext = bytearray()
        ext_for = {}
        ext_base = header + ifd_size
        for t in order:
            typ, cnt, vals = tags[t]
            vsize = TYPESIZE.get(typ, 1) * cnt
            if vsize <= 4:
                continue
            if t == TAG_STRIPOFFSETS:
                vals = soffs
            if typ == 3:
                data = struct.pack("<%dH" % cnt, *vals)
            elif typ == 4:
                data = struct.pack("<%dI" % cnt, *vals)
            elif typ == 5:
                data = b"".join(struct.pack("<II", a, bb) for a, bb in vals)
            elif typ == 2:
                data = bytes(vals)
            else:
                data = bytes(vals)
            ext_for[t] = ext_base + len(ext)
            ext += data
            if len(ext) % 2:
                ext += b"\x00"
        return ext, ext_for

    # Strip data follows header + IFD + external blocks. External block size is
    # independent of the offset values themselves (counts are fixed), so one
    # build pass is enough to learn the base, then we compute final offsets.
    ext0, _ = build([0] * nstrips)
    strip_base = header + ifd_size + len(ext0)
    soffs = []
    cur = strip_base
    for c in counts:
        soffs.append(cur)
        cur += c
    tags[TAG_STRIPOFFSETS][2] = soffs
    ext, ext_for = build(soffs)
    assert header + ifd_size + len(ext) == strip_base

    out = bytearray()
    out += b"II"
    out += struct.pack("<H", 42)
    out += struct.pack("<I", 8)
    out += struct.pack("<H", ncnt)
    for t in order:
        typ, cnt, vals = tags[t]
        vsize = TYPESIZE.get(typ, 1) * cnt
        out += struct.pack("<HHI", t, typ, cnt)
        if vsize <= 4:
            if typ == 3:
                vb = struct.pack("<%dH" % cnt, *vals)
            elif typ == 4:
                vb = struct.pack("<%dI" % cnt, *vals)
            elif typ == 2:
                vb = bytes(vals)
            else:
                vb = bytes(vals)
            out += vb + b"\x00" * (4 - len(vb))
        else:
            out += struct.pack("<I", ext_for[t])
    out += struct.pack("<I", 0)
    out += ext
    for e in enc:
        out += e
    open(path, "wb").write(out)


# ---------------------------------------------------------------------------
# Geometry solve: comp_x = s*ref_x + t_y  (global scale, per-line shift)
# ---------------------------------------------------------------------------

def _row_ncc_peak(a, b, maxlag=20):
    """Return (peak_value, lag) of normalized cross-correlation.

    Lag is constrained to |lag| <= maxlag (per-line jitter is small in TBC
    output, and the global argmax otherwise locks onto spurious far peaks).
    Positive lag means b is shifted right relative to a. Sub-pixel refined.
    """
    a = a - a.mean()
    b = b - b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0, 0.0
    n = len(a)
    fsize = 1
    while fsize < 2 * n:
        fsize *= 2
    fa = np.fft.rfft(a, fsize)
    fb = np.fft.rfft(b, fsize)
    corr = np.fft.irfft(fa * np.conj(fb), fsize)
    corr = np.concatenate((corr[-(n - 1):], corr[:n])) / (na * nb)
    lags = np.arange(-(n - 1), n)
    mask = np.abs(lags) <= maxlag
    sub = corr[mask]
    sublags = lags[mask]
    k = int(np.argmax(sub))
    peak = sub[k]
    lag = float(sublags[k])
    # parabolic sub-pixel refinement
    if 0 < k < len(sub) - 1:
        y0, y1, y2 = sub[k - 1], sub[k], sub[k + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-12:
            lag = lag + 0.5 * (y0 - y2) / denom
    return float(peak), float(lag)


def _resample_width(plane, scale):
    """Resample each row at columns scale*j (sub-pixel) onto ref width."""
    H, W = plane.shape
    cols = scale * np.arange(W)
    cols = np.clip(cols, 0, W - 1)
    rows = np.arange(H)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return ndimage.map_coordinates(plane, [rr, cc], order=1, mode="nearest")


def solve_geometry(ref_Y, comp_Y, global_shift=False):
    """Find global scale s and per-line shift t_y aligning comp to ref luma."""
    Href, Wref = ref_Y.shape
    Hc, Wc = comp_Y.shape

    # comp sampled at column s*j+t maps onto ref column j. The luma content of
    # comp spans Wc; ref spans Wref. Seed scale from width ratio.
    seed = Wc / Wref
    rows_sample = np.arange(0, Href, max(1, Href // 64))

    def neg_score(s):
        total = 0.0
        # resample comp rows at step s, compare against ref rows
        cols = s * np.arange(Wref)
        cols = np.clip(cols, 0, Wc - 1)
        for y in rows_sample:
            rrow = ref_Y[y]
            crow = ndimage.map_coordinates(
                comp_Y[y][None, :], [np.zeros_like(cols), cols],
                order=1, mode="nearest")
            peak, _ = _row_ncc_peak(rrow, crow)
            total += peak
        return -total

    lo, hi = max(0.3, seed * 0.6), min(3.0, seed * 1.6)
    res = optimize.minimize_scalar(neg_score, bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-3})
    s = float(res.x)

    # Per-line shift at optimal s. The NCC lag is measured on the ref grid
    # (positive = comp content sits to the right); apply_geometry samples comp
    # at column s*j + t, so the comp-pixel offset is t = -s*lag. Gate on peak
    # quality: rows whose correlation is weak (noise/black lines) keep t=0
    # rather than injecting a spurious jump.
    cols = s * np.arange(Wref)
    cols = np.clip(cols, 0, Wc - 1)
    zeros = np.zeros_like(cols)
    t = np.zeros(Href)
    lags = []
    for y in range(Href):
        crow = ndimage.map_coordinates(
            comp_Y[y][None, :], [zeros, cols], order=1, mode="nearest")
        peak, lag = _row_ncc_peak(ref_Y[y], crow)
        if peak > 0.5:
            t[y] = -s * lag
            lags.append(lag)
    if global_shift:
        # Collapse to one shift for every row: median lag over the rows that
        # correlated well (robust to the gated-out noise/black lines).
        glag = float(np.median(lags)) if lags else 0.0
        t = np.full(Href, -s * glag)
    return s, t


def solve_geometry_chroma(ref_U, ref_V, comp_U, comp_V, s, global_shift=False):
    """Per-line chroma shift at fixed scale s (shared U+V correlation)."""
    Href, Wref = ref_U.shape
    Wc = comp_U.shape[1]
    cols = s * np.arange(Wref)
    cols = np.clip(cols, 0, Wc - 1)
    zeros = np.zeros_like(cols)
    t = np.zeros(Href)
    lags = []
    for y in range(Href):
        cu = ndimage.map_coordinates(comp_U[y][None, :], [zeros, cols],
                                     order=1, mode="nearest")
        cv = ndimage.map_coordinates(comp_V[y][None, :], [zeros, cols],
                                     order=1, mode="nearest")
        pu, lu = _row_ncc_peak(ref_U[y], cu)
        pv, lv = _row_ncc_peak(ref_V[y], cv)
        denom = pu + pv
        # Quality-weighted shared lag (ref grid); gate weak rows to t=0.
        if denom > 1e-9 and (pu > 0.5 or pv > 0.5):
            lag = (pu * lu + pv * lv) / denom
            t[y] = -s * lag
            lags.append(lag)
    if global_shift:
        glag = float(np.median(lags)) if lags else 0.0
        t = np.full(Href, -s * glag)
    return t


def apply_geometry(plane, s, t, out_shape):
    """Resample plane: output (y,j) samples plane at (y, s*j + t[y])."""
    oh, ow = out_shape
    H, W = plane.shape
    j = np.arange(ow)
    cols = s * j[None, :] + t[:, None]
    cols = np.clip(cols, 0, W - 1)
    rows = np.repeat(np.arange(oh)[:, None], ow, axis=1).astype(np.float64)
    rows = np.clip(rows, 0, H - 1)
    return ndimage.map_coordinates(plane, [rows, cols], order=3, mode="nearest")


# ---------------------------------------------------------------------------
# Value solve: ref = g*comp + off  (global, inlier least squares, clip 0-255)
# ---------------------------------------------------------------------------

def solve_values(ref, comp):
    """Least-squares (g, off) over non-clipped inlier pixels."""
    r = ref.reshape(-1)
    c = comp.reshape(-1)
    mask = (r > 0) & (r < 255) & (c > 0) & (c < 255)
    if mask.sum() < 16:
        mask = np.ones_like(r, dtype=bool)
    A = np.vstack([c[mask], np.ones(mask.sum())]).T
    (g, off), *_ = np.linalg.lstsq(A, r[mask], rcond=None)
    return float(g), float(off)


def apply_values(plane, g, off):
    return np.clip(np.rint(g * plane + off), 0, 255)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv):
    args = [a for a in argv[1:] if not a.startswith("-")]
    flags = {a for a in argv[1:] if a.startswith("-")}
    global_shift = "--global-shift" in flags
    flags.discard("--global-shift")
    if len(args) != 3 or flags:
        sys.stderr.write(
            "usage: python lineup.py [--global-shift] REF COMP OUT\n")
        return 2
    ref_path, comp_path, out_path = args

    try:
        ref = read_tiff(ref_path)
        comp = read_tiff(comp_path)
    except ValueError as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    if ref["H"] != comp["H"]:
        sys.stderr.write(
            f"error: heights differ (ref {ref['H']} vs comp {comp['H']}); "
            "only horizontal differences are supported\n")
        return 1

    # Geometry on luma.
    s, t_y = solve_geometry(ref["Y"], comp["Y"], global_shift)
    out_shape = (ref["H"], ref["W"])
    Yg = apply_geometry(comp["Y"], s, t_y, out_shape)

    # Geometry on chroma: shared scale, own shift.
    t_c = solve_geometry_chroma(
        ref["U"], ref["V"], comp["U"], comp["V"], s, global_shift)
    Ug = apply_geometry(comp["U"], s, t_c, out_shape)
    Vg = apply_geometry(comp["V"], s, t_c, out_shape)

    # Value solve.
    gy, offy = solve_values(ref["Y"], Yg)
    Yout = apply_values(Yg, gy, offy)

    # Shared chroma value transform pooling U+V inliers.
    rc = np.concatenate([ref["U"].reshape(-1), ref["V"].reshape(-1)])
    cc = np.concatenate([Ug.reshape(-1), Vg.reshape(-1)])
    gc, offc = solve_values(rc, cc)
    Uout = apply_values(Ug, gc, offc)
    Vout = apply_values(Vg, gc, offc)

    write_tiff(out_path, ref, Yout, Uout, Vout)
    mode = "global" if global_shift else "per-line"
    sys.stderr.write(
        f"lineup: scale={s:.5f} shift={mode} "
        f"luma(g={gy:.4f},off={offy:.2f}) "
        f"chroma(g={gc:.4f},off={offc:.2f}) -> {out_path}\n")
    return 0


def _cli():
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    _cli()
