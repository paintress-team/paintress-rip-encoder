# Getting good prints out of `rip.py`

`rip.py` is the first step of Paintress. It turns an image into a **RIP
payload** (a `.json` header and a packed `.bin`), which the encoder then turns
into a print job for the daemon and the firmware.

```
image ─► rip.py ─► RIP payload (.json + .bin) ─► encoder.py ─► job ─► daemon ─► firmware
```

This guide goes through every option of `rip.py` and the order of work that
gives the best prints. What makes the biggest difference is **measuring your
setup** (steps 1 to 4 below). A print with the default profile is only a
starting point.

## In short

```sh
# Once per media and DPI: calibrate
python rip/rip.py --target nozzle_check --dpi 630 -o out/nz.json --preview out/nz.png
#   ...print it, read the dead nozzles, then:
python tools/record_dead_nozzles.py --dead "M:4,7;Y:14,19;K:58" \
       --profile rip/profiles/mymedia.json --out rip/profiles/mymedia.json

# Once per head: measure the distance between the colour columns (X alignment)
python rip/rip.py --target col_align --dpi 630 -o out/colal.json --preview out/colal.png
#   ...encode with the default gaps and print, read the straightest k for each ink, then:
python tools/col_align_gaps.py "C:-2;Y:3;K:5" --payload out/colal.json
#   -> prints the corrected --column-gap / --group-gap for the ENCODER

python rip/rip.py --target wedge --dpi 630 -o out/wedge.json --preview out/wedge.png
#   ...print it, scan it (about 300 spi), then:
python tools/calibrate_from_scan.py wedge_scan.png --payload out/wedge.json \
       --profile rip/profiles/mymedia.json --out rip/profiles/mymedia.json --gray-balance

# Every real print after that: point at the media profile
python rip/rip.py photo.tif -o out/job.json --dpi 630 --calibration rip/profiles/mymedia.json
python rip/viewer.py out/job.json -o out/chan/ --soft-proof out/proof.png   # preview and ink estimate
```

## 1. The steps

Good prints come from calibrating your actual **media, ink and DPI**, not from
tweaking options. Do steps 1 to 4 once. The column alignment (step 2) is done
**once per head**, the others **once per media** (and per DPI you print at).
After that, step 5 is one command per job.

### Step 0: pick a head and a DPI

`--head` picks the layout: `c6n90` (the default, six columns of 90 nozzles)
or `c4n180` (two columns of 180: black alone on the left, and C, M and Y as
three blocks of 60 on the right). `--ink-map` says which ink is in which slot,
in the order you read the head facing it. By default it's the head's standard
setup, or the `ink_map` in your calibration profile.

The DPI must be a **multiple of the head's nozzle pitch** (90 on `c6n90`, 180
on `c4n180`), so that the number of passes per band is a whole number. A
higher DPI gives finer dots and smoother gradients, but more passes, so it's
slower and uses more ink. `--list-dpi` lists the values for the chosen head.

| DPI  | Passes per band | Use |
|------|-----------------|-----|
| 360  | 4   | fast, drafts |
| 630  | 7   | **a good default** |
| 720  | 8   | high quality |
| 900+ | 10+ | very fine, slow |

Calibrate at the **same DPI** you'll print at: the tone curves are stored per
DPI.

### Step 1: nozzle check

```sh
python rip/rip.py --target nozzle_check --dpi 630 -o out/nz.json --preview out/nz.png
# encode and print out/nz.json, then read it by eye
```

The printed check is a set of **combs with their own labels**: one comb per
ink that is connected (six on `c6n90`, four on `c4n180`), one above the other.
Each nozzle fires one dash, the dashes step along a diagonal, there's a
**numbered ruler** on top, and **lines** run down the page every 10 nozzles.
Only the nozzles a print really fires are checked. A **gap is a dead or
clogged nozzle**. To read it: find the gap, follow the line up to the ruler,
read the number. Read the **number** printed there; don't guess from the
position.

Record the dead nozzles:

```sh
python tools/record_dead_nozzles.py --dead "C:89;M:4,7,64;Y:14,19;K:58,73" \
       --profile rip/profiles/mymedia.json --out rip/profiles/mymedia.json
```

> **Clog or dead nozzle?** Several dead nozzles in a row (for example
> `M:70,71,72,73`) are almost always a **clog**. **Purge and print the check
> again** before you compensate. `retouch` can fix single dead nozzles and the
> two *edges* of a group, but the **middle** of a group has no working
> neighbour to borrow from, and software can't fix it. Purge it.

### Step 2: column alignment (colour to colour)

The encoder lines the colours up by delaying each channel by its real
distance behind the leading column (M), set with `--column-gap` and
`--group-gap`. The defaults are already the measured `c6n90` values (3 and 25
nozzle pitches of 1/90", which gives 0.842 and 7.051 mm, rounded down a hair
because of how the encoder rounds). Do this step again if the head is
remounted, if the sweep speed changes a lot, or for a different head. If the
values are wrong, **every colour lands shifted in X by the same amount, on
every job**. This target measures the real distances:

```sh
python rip/rip.py --target col_align --dpi 630 -o out/colal.json --preview out/colal.png
# encode it with the gaps you want to check (the encoder's defaults) and print
```

For each tested ink (C, Y, K) the print has a row of cells. Each cell is a
magenta mark, a mark of the tested ink and another magenta mark, stacked,
with the ink mark shifted by `k` pixels (the ruler shows `k`). Where the three
marks are **in a perfect line**, that shift cancelled the error. The cells are
built so that trigger jitter and a head tilt don't affect them: all three
marks come from the same band, and the ink mark is centred between the two
magenta ones. A **tilt band** at the bottom prints one full-height line per
column. How much it leans is the head's tilt; how much it wiggles is how
straight the column is.

With a loupe, find the straightest cell for each ink, read its `k` on the
ruler, then:

```sh
python tools/col_align_gaps.py "C:-2;Y:3;K:5" --payload out/colal.json
```

The tool prints each ink's error, the real gaps that fit best, and the
encoder flags to use from now on, for example
`--column-gap 0.874 --group-gap 7.031`. These are **encoder** flags; the RIP
has no gap options.

The gaps are real millimetres, so **one measurement works for every DPI and
media**. Only measure again if the head is remounted or the sweep speed
changes a lot (the time a drop takes to fly changes the distance it seems to
have). If no cell looks straight anywhere, the gaps are further off than the
row covers: print it again with `--span-mm 2 --cell-pitch-mm 5`.

> On `c4n180` the fit isn't right yet: `col_align_gaps.py` only knows the
> `c6n90` column layout. See Known issues in the README.

### Step 3: tone wedge and gray balance

```sh
python rip/rip.py --target wedge --dpi 630 -o out/wedge.json --preview out/wedge.png
# encode and print, then scan the printed sheet
```

Scan the wedge with **automatic exposure, automatic colour, sharpening and
descreen turned off**, more or less upright, at about **300 spi** (600 works
too). What matters is using the same settings every time, not exact colour.
Then:

```sh
python tools/calibrate_from_scan.py wedge_scan.png --payload out/wedge.json \
       --profile rip/profiles/mymedia.json --out rip/profiles/mymedia.json \
       --gray-balance --preview out/wedge_qa.png
```

This measures the printed tone of each channel and writes **measured tone
curves** into the profile (`dpi_curves[<dpi>]`). With `--gray-balance` it
also adjusts C, M and Y so **grays print gray**, at the cost of a little
saturation. Try it with and without and keep what looks best. Use `--preview`
to check that the sampling boxes are inside the patches. The tool also prints
how far the gray row is from a straight line (an RMSE): a number to follow
from one try to the next (lower is better).

### Step 4 (optional): set a drop size for DPIs you haven't measured

If you print at a DPI you haven't measured, set `drop_diameter_um` in the
profile (one number per media). The RIP then builds a **modelled** tone curve
(Murray-Davies) instead of falling back to the rough default. The simulated
print and the ink estimate use it too.

### Step 5: print

```sh
python rip/rip.py photo.tif -o out/job.json --dpi 630 --calibration rip/profiles/mymedia.json
```

Tips:
- **Use a 16-bit image** (TIFF or PNG) if you have one. The RIP keeps 16-bit
  precision and resizes in linear light, so gradients don't band.
- Keep `--dither floyd_steinberg`. It gives the sharpest detail.
  `blue_noise` and `ordered` are other options, not better ones.
- Add `--nozzle-comp retouch` if you have **single** dead nozzles (complete,
  about twice the sweeps). `reroute` is a free, partial fix.
- Add `--shingle 2` on media that **doesn't absorb ink** (plastic, glass), so
  wet drops settle. Add `--band-overlap 12` if you see a **seam repeating at
  every band**.
- Set the printed size with `--width` / `--height` (mm) if you need to.

### Step 6: check before printing

```sh
python rip/viewer.py out/job.json -o out/chan/ --soft-proof out/proof.png
```

This draws a **simulated print** (drops spread out and blended) that shows the
tone, the grain, streaks from dead nozzles and band seams, and prints an
**ink estimate** (µL per ink). It's a quick check before a long print.

## 2. All the options of `rip.py`

`python rip/rip.py -h` shows the built-in help. The **calibration profile
decides**: each colour flag below changes one field of the loaded profile, in
memory, for this run only.

### Input and output

| Option | Default | What it does |
|--------|---------|--------------|
| `input` | required | The image (PNG, JPG, TIFF, 8 or 16 bit, RGB, RGBA, gray or CMYK). Leave it out only with `--target`. |
| `-o`, `--output` | required | The `.json` header to write; the packed `.bin` goes next to it. |
| `--list-dpi` | | List the DPI values that work and exit (follows `--head`). |

### Head and ink setup

| Option | Default | What it does |
|--------|---------|--------------|
| `--head {c6n90,c4n180}` | `c6n90` | The head. It decides the channels, the nozzle pitch (so the DPI values that work), the slots and the band step. |
| `--ink-map` | the head's standard, or the profile's `ink_map` | Which ink is in which slot, separated by commas, in the order you read the head facing it (`-` = empty). The standard setup is `K,Y,LM,LC,C,M` on `c6n90` and `K,C,M,Y` on `c4n180`. |

### Resolution and size

| Option | Default | What it does |
|--------|---------|--------------|
| `--dpi N` | the head's own (630 on `c6n90`, 720 on `c4n180`) | Print resolution. **It must be a multiple of the head's nozzle pitch** (90 npi on `c6n90`, 180 on `c4n180`). Higher is finer and slower. `--list-dpi` lists the values. |
| `--width MM` | the image's own | Resize the image to this printed width (mm) at the DPI. |
| `--height MM` | the image's own | Resize to this printed height (mm). Give one of the two to keep the proportions; give both to force them (the image can stretch). |

### Halftoning

| Option | Default | What it does |
|--------|---------|--------------|
| `--dither {floyd_steinberg,blue_noise,ordered}` | floyd_steinberg | How dots are placed. Floyd-Steinberg is the sharpest and the one to use. |
| `--blue-noise-size {32,64,128,256}` | 64 | Size of the blue-noise texture (only with `--dither blue_noise`). |

### Colour and calibration

| Option | Default | What it does |
|--------|---------|--------------|
| `--calibration PATH` | the default profile | The calibration profile JSON for the media. **It decides.** |
| `--icc PATH` | none | A CMYK ICC profile for the RGB to CMYK conversion (through `PIL.ImageCms`). Without it, a UCR/GCR conversion is used. |
| `--dot-gain V` | the profile's | Dot gain, `0..1` (higher makes the mid-tones lighter). Set as C=M=V, Y=0.75V, K=1.25V. *Only used when there is no tone curve.* |
| `--no-dot-gain` | off | Turn dot gain off. |
| `--ink-scale V` | the profile's | Multiplies all ink before halftoning. |
| `--cyan-scale V` | the profile's | Multiplies cyan (and LC). Lower it if prints look green or cyan. |
| `--magenta-scale V` | the profile's | Multiplies magenta (and LM). Raise it if prints look green. |
| `--yellow-scale V` | the profile's | Multiplies yellow. Lower it if prints look green or yellow. |
| `--black-scale V` | the profile's | Multiplies black. |
| `--max-total-ink V` | the profile's | The most CMYK ink on one pixel (a fraction), applied after the tone curves. |
| `--black-generation V` | the profile's | How much black replaces CMY, `0..1` (without `--icc`). |
| `--black-start V` | the profile's | Where black starts, `0..1`: grays below it stay CMY (no grainy black in light areas), above it black comes in. |
| `--reference-dpi N` | the profile's | The DPI where the automatic ink reduction starts. |
| `--no-dpi-compensation` | off | Don't reduce ink above the reference DPI. |
| `--no-ink-limit` | off | No ink limit and no per-channel scaling at all. |

> **Which colour path is used?** If the profile has measured curves for all
> of C, M, Y and K at the print's DPI (`dpi_curves`), the RIP uses **those
> curves and the total ink limit, nothing else**: they already include dot
> gain, DPI reduction and the per-channel amounts (and gray balance, if you
> measured it). If not, and `drop_diameter_um` is set, it uses a **modelled**
> curve. If not, it uses the **rough default** (dot gain, per-channel amounts,
> DPI reduction and the ink limit). The job stores which one it used in
> `metadata.processing.colour_mode` and `lut_source`.

### Print structure

| Option | Default | What it does |
|--------|---------|--------------|
| `--nozzle-comp {none,reroute,retouch}` | none | Hide the profile's `dead_nozzles`. `reroute` moves their ink to the rows next to them during Floyd-Steinberg (free, partial; in a large group the ink that can't reach a working row is dropped, so it smears locally instead of far away). `retouch` adds passes so the **nearest working nozzle** (`n-1`, else `n+1`) prints the dead rows again (complete, about twice the sweeps). It fixes single nozzles and both **edges** of a group; the **middle** of a group (both neighbours dead) can't be fixed and is reported and left as is. |
| `--shingle N` | the profile's, or 1 | For media that doesn't absorb ink: print each pass in N sweeps that alternate columns, so wet drops settle. 1 = off, 2 = even/odd. Takes N times as many sweeps. |
| `--band-overlap N` | the profile's, or 0 | Overlap bands by N rows and split the seam at random, to hide a seam that repeats at every band. 0 = off. |

### Calibration targets (no image)

`--target` makes a target instead of processing an image.

| Option | Default | What it does |
|--------|---------|--------------|
| `--target {wedge,nozzle_check,col_align}` | | `wedge` = step wedge for the tone curves and gray balance; `nozzle_check` = fires every nozzle once; `col_align` = distance between the colour columns (step 2). One target per DPI. |
| `--steps N` | 33 | Number of levels per channel in the wedge. |
| `--max-x-mm MM` | 200 | Largest wedge width (the direction the head sweeps). |
| `--max-y-mm MM` | 200 | Largest wedge height (the direction the paper advances). |
| `--gap-mm MM` | 2.0 | Space between wedge patches. |
| `--span-mm MM` | 0.8 | col_align: how far the cells shift, ± this many mm, in whole pixels. |
| `--cell-pitch-mm MM` | 2.5 | col_align: X distance between cells. |
| `--mark-mm MM` | 0.5 | col_align: width of a mark. |
| `--repeats N` | 2 | col_align: how many times the row of cells is repeated. |
| `--dash-mm MM` | 1.5 | Nozzle check: length of a dash (in X). |
| `--dx-mm MM` | 1.8 | Nozzle check: X step from one nozzle's dash to the next. |
| `--preview PATH` | none | Also write a labelled RGB preview PNG of the target. |

## 3. The calibration profile (JSON)

The profile decides everything about colour. The one in the repository,
`rip/profiles/default.json`, is kept **generic**: don't save measurements into
it. Make one profile per media and pass it with `--calibration`. The fields:

```jsonc
{
  "media": "my-glossy-vinyl",           // a name, any text
  "head": "c6n90",                       // optional: the head it was measured on
  "dot_gain": {                          // only used when there is no tone curve
    "enabled": true,
    "C": 0.50, "M": 0.50, "Y": 0.375, "K": 0.625
  },
  "channel_scales": {                    // rough gray balance / strength of each ink
    "C": 0.55, "M": 0.80, "Y": 0.45, "K": 0.55, "LC": 0.55, "LM": 0.80
  },
  "global_scale": 1.0,                   // multiplies all ink
  "max_total_ink": 0.90,                 // most ink per pixel (fraction), after the tone curves
  "dpi_compensation": {                  // rough ink reduction above reference_dpi
    "enabled": true, "reference_dpi": 360, "power": 2.0
  },
  "black_generation": 1.0,               // how much black replaces CMY (without --icc)
  "black_start": 0.0,                    // where black starts (0 = from the lightest tones)
  "drop_diameter_um": null,              // drop size: modelled tone curve, simulated print, ink estimate
  "ink_map": ["K","Y","LM","LC","C","M"],// ink in each slot, left to right
  "dead_nozzles": {},                    // per slot, e.g. {"0": [58], "5": [4,7]}
  "shingle_passes": 1,                   // 1 = off
  "band_overlap": 0,                     // 0 = off
  "dpi_curves": {}                       // measured tone curves per DPI: {"630": {"C":[256],"M":[256],"Y":[256],"K":[256]}}
}
```

- **`dpi_curves`** matters most for quality. These are the measured curves
  written by `calibrate_from_scan.py`. When there is a set for the print's
  DPI, they replace the rough default.
- **`ink_map`** is the ink setup: which ink feeds each slot, left to right. It
  lives here only: the RIP writes it into every payload (the encoder uses it
  from there) and uses it to store `dead_nozzles` per slot.
- **`head`** is optional and names the head the profile was measured on. If
  it's set and doesn't match `--head`, the RIP **stops with an error**:
  `dead_nozzles` are stored per slot and `ink_map` describes one head's setup,
  so neither works on another head. Leave it out and there's no check.
- **`dead_nozzles`** is what `--nozzle-comp` uses. `record_dead_nozzles.py`
  writes it, per **slot** (not per ink), so it stays right if you connect the
  inks differently; the RIP works out the channels through `ink_map`. It
  covers every connected ink, light inks too.
- The `--*-scale`, `--dot-gain`, `--black-*`, `--shingle`, `--band-overlap`,
  `--max-total-ink` and `--reference-dpi` flags each change one field for one
  run, without touching the file.

## 4. The other tools

### `tools/calibrate_from_scan.py`: build the tone curves

| Option | Default | What it does |
|--------|---------|--------------|
| `scan` | required | The scan of the printed wedge. |
| `--payload PATH` | required | The wedge payload that was printed (it holds the layout). |
| `--profile PATH` | required | The profile to update. |
| `--out PATH` | the same file | Where to write the updated profile. |
| `--inset F` | 0.3 | How much of each patch's edge to leave out before averaging. |
| `--gray-balance` | off | Also balance gray: make C, M and Y end at the same darkness so grays print gray. |
| `--preview PATH` | none | A check image (corner squares and sampling boxes). |
| `--force` | off | Save the curves even if they look broken. |

It finds the target on the scan from its four corner squares (a slight tilt
is fine), measures each patch in the colour its ink absorbs, and turns the
tone curve into a lookup table. A few clogged nozzles don't throw it off: it
takes the median of the row averages, and it refuses to save a curve that
looks broken. Calibrate **one DPI per scan**.

### `tools/record_dead_nozzles.py`: record dead nozzles

| Option | Default | What it does |
|--------|---------|--------------|
| `--dead "C:3,17;M:42"` | required | Dead nozzles per ink, read off the printed check. An empty string clears the list. |
| `--profile PATH` | required | The profile to update. |
| `--out PATH` | the same file | Where to write it. |

### `tools/col_align_gaps.py`: work out the column gaps

| Option | Default | What it does |
|--------|---------|--------------|
| `readings` | required | The straightest cell's `k` for each ink, read on the printed ruler, for example `"C:-2;Y:3;K:5"`. |
| `--payload PATH` | required | The col_align payload that was printed. |
| `--column-gap MM` | 0.842 | The `column_gap_mm` the target was **encoded** with (same default as the encoder). |
| `--group-gap MM` | 7.051 | The `group_gap_mm` the target was **encoded** with (same default as the encoder). |
| `--json PATH` | none | Also write the full report as JSON. |

It takes each ink's error as `-k`, works out the exact pixel offsets the
encoder used, gives each ink's **real distance behind magenta**, and finds the
two gaps that fit best. How far each ink is from the fit tells you whether two
evenly spaced gaps really describe the head. Read the head's tilt and the
straightness of its columns by eye on the tilt band.

### `rip/viewer.py`: look at a job and simulate the print

| Option | Default | What it does |
|--------|---------|--------------|
| `input` | required | A RIP payload `.json`. |
| `-o`, `--output-dir` | ./output_channels | Where the images go. |
| `--soft-proof PATH` | none | A simulated print (drops spread out and blended). |
| `--drop-diameter-um V` | the profile's, or 40 | Drop size for the simulated print and the ink estimate. |
| `--drop-volume-pl V` | estimated from the size | Measured drop volume for the ink estimate. |
| `--format {png,tiff,bmp}` / `--no-invert` / `--no-combined` | | Control the images it writes. |

It also prints an **ink estimate** per ink (dots fired × drop volume) for
every payload.

## 5. Checklist

1. **Purge the head** and print a **nozzle check** first. Fix clogs on the
   head; only record *single* dead nozzles.
2. **Align the columns once per head** (`col_align`). The corrected
   `--column-gap` / `--group-gap` on the encoder remove the fixed X offset
   between colours.
3. **Calibrate a wedge** for your media at the DPI you'll print at, with
   `--gray-balance`. This makes the biggest difference.
4. Print from a **16-bit** image and let the RIP do the resizing.
5. Keep **Floyd-Steinberg** dithering.
6. Use `--nozzle-comp retouch` for single dead nozzles.
7. Media that doesn't absorb ink: `--shingle 2`. A seam you can see between
   bands: `--band-overlap`.
8. **Simulate the print** before printing, and look at the ink estimate.
9. Calibrate the wedge again when the media, ink or DPI changes, and watch the
   RMSE go down.

## 6. Good to know

- **One target per DPI**, for the wedge and for the nozzle check. The
  col_align result is in **mm** and works at every DPI, but the target itself
  still has to be made at a DPI the head supports (a multiple of its nozzle
  pitch). A higher DPI gives smaller steps between cells.
- Keep the **default profile generic**. Never save measured curves into
  `rip/profiles/default.json`; use `--out` with a profile for your media.
- The RIP prints **the right way round** (not mirrored), so a scan placed
  more or less upright matches the wedge layout.
- A group of dead nozzles next to each other **can't** be fixed in software.
  Purge it.
- After `rip.py`, run the **encoder** to get the job you can print:
  `python encoder/encoder.py out/job.json -o out/print.json`.
- If your inks aren't connected the head's standard way (`K, Y, LM, LC, C, M`
  on `c6n90`, `K, C, M, Y` on `c4n180`, left to right facing the head), set
  `ink_map` in the calibration profile. The RIP writes it into the payload and
  the encoder uses it from there; `--ink-map` (for example
  `--ink-map "K,Y,-,-,C,M"` for four inks, `-` = empty slot) overrides it.
  `record_dead_nozzles.py` stores dead nozzles per slot through the same map,
  so everything agrees on which ink is where.
