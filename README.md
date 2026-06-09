# lineup

Warp a comparand TIFF onto a reference TIFF's grid so the two can be diffed side-by-side. The images are assumed to be the same content differing only by a global horizontal stretch plus a horizontal shift, and a global value (luma/chroma) shift+stretch. There are no vertical changes; every row is solved independently.

## Install

```
pip install .
```

This pulls in `numpy` and `scipy` and installs a `lineup` command. To work on the code, install editable with `pip install -e .`.

## Usage

The program only supports 8-bit YCbCr 4:2:2 TIFFs of equal height. Since you probably have videos on hand, here's an example command to extract frames with `ffmpeg` from an interlaced source, displaying as stacked fields:

```
ffmpeg -ss 00:12:34.560 -i video.mkv \
  -filter_complex "[0:v]separatefields,split=2[a][b];[a]select='not(mod(n\,2))',setpts=PTS-STARTPTS[top];[b]select='mod(n\,2)',setpts=PTS-STARTPTS[bottom];[top][bottom]vstack=inputs=2,format=yuv422p[v]" \
  -map "[v]" -t 1 -an \
  -fps_mode passthrough \
  -c:v tiff \
  -pix_fmt yuv422p \
  -f image2 "frames_%06d.tif"
```

This will provide you with a handful of TIFFs. Perform the same on the other source video, select visually matching frames, then run `lineup` on them:

```
lineup [--global-shift] REF COMP OUT
```

(or `python lineup.py [--global-shift] REF COMP OUT` without installing)

- `REF` — reference TIFF. Defines the output grid and TIFF specs.
- `COMP` — comparand TIFF. Warped to line up with `REF`.
- `OUT` — output TIFF. The transformed comparand, written with `REF`'s exact IFD (dimensions, subsampling, compression, resolution, ReferenceBlackWhite, RowsPerStrip, etc.).
- `--global-shift` — apply a single horizontal shift to every row instead of a per-line shift (see below).

Example:

```
python lineup.py decode.tif conventional.tif conventional_aligned.tif
```

For in-browser display, you can convert the files to AVIF with no fidelity loss:

```
ffmpeg -i conventional_aligned.tif -pix_fmt + -c:v libaom-av1 -crf 0 -b:v 0 -cpu-used 0 conventional_aligned.avif
```

After this, you can use online image diff tools with the outputs. [Example](https://imgcmp.com/compare#H4sIAA1BKGoAA7WRQYvCQAyF_0qZszR67XV3WdjLHtybFInT2A5MM2WScWUX_7tjK6Lg0d5CeC9f8vJv1KknU5mf1EqxXq4-Vsuq-HJ8pMYsjIQULYmpNmY9lrn3rR3FAr1redR8-rBDf2vU2WWJJ9Nb4AOxusDos_SdbGjoIknRXwQb06kOUgFE_C1bp13aJaFoA2v2lTb0wNjjnwTwjikNEGkv0BE2Aj2KUgQ6Yj94gmy6wUo8uH0mvnB8My6_vd45O2DbjrlOnHrx0qQm0gwn3L9gxqSeYR7zqk9n3B1J0dsCAAA).

## What it does

1. **Geometry.** Models the mapping as `comp_x = s*ref_x + t_y`: one global horizontal scale `s` (found by maximizing summed per-row normalized cross-correlation), plus a horizontal shift `t_y` per row from sub-pixel NCC. The comparand is resampled onto the reference grid with cubic interpolation.
2. **Value.** Fits one global `ref = g*comp + off` by least squares over non-clipped inlier pixels, then applies it and clips to 0-255. Crushing of blacks/whites emerges naturally from the clip. Luma gets its own `(g, off)`; chroma U and V share one pooled `(g, off)`.

Chroma inherits luma's global scale and solves only its own shift and value transform; U and V receive identical parameters.

### Shift modes

- **Per-line (default).** Each row gets its own shift, tracking horizontal jitter line by line. Rows that correlate poorly (noise, black lines) are left unshifted rather than jumped to a spurious position.
- **`--global-shift`.** The per-line shifts are collapsed to a single median shift (robust to the gated-out rows) applied uniformly to every row.

## I/O format

Self-contained little-endian TIFF reader/writer for 8-bit YCbCr (Photometric 6), chunky `[Y0,Y1,Cb,Cr]` 4:2:2, PackBits compression. Internally all planes are processed as full-width float64 (chroma spline-upsampled in, spline-downsampled out). The writer clones the reference IFD verbatim and re-encodes PackBits per scanline, so the output matches the reference's specs by construction.

The tool exits non-zero with a clear message if an input is not YCbCr or if the two images differ in height. A non-4:2:2 subsampling prints a warning and falls back to a best-effort spline upsample.

## Requirements

Python >= 3.9. Dependencies (`numpy`, `scipy`) are declared in `pyproject.toml` and installed automatically by `pip install .`.

## License

BSD Zero Clause License. See `LICENSE` for details. This whole thing was vibecoded anyway, so do whatever you want with it.
