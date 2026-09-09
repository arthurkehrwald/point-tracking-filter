# Point tracking filter

This project is about processing 3D point tracking data of an infrared LED recorded using two separate camera systems.
The data is loaded from CSV files. The recordings from the Luxonis Oak-D stereo camera contain only the positions of the
LED over time. The Optitrack recordings also contain the positions of three markers attached to the Oak-D camera in a
rigid formation. This makes it possible to align the reference frames.

## Running

```
uv sync
uv run point-tracking-filter
```

Select recordings in the browser on the left and press "Show in player". Recordings are paired by the leading token of
their file name (`slow2`, `sweep`, ...), and their time axes are normalized so that the first appearance of the LED is
`t = 0`.

The transform into the camera's reference frame is not fitted from the trajectory data. Instead the positions of the
three tracking markers are measured by hand in the Oak-D camera's own coordinate space — the origin is the optical
center and, seen from the camera, `+x` points right, `+y` up and `+z` forward — and stored in `config/extrinsic.toml`
(editable in the analysis panel).

Because the markers form an irregular triangle, the first three coordinate triplets of an Optitrack recording can be
assigned to them automatically: the side lengths of the recorded triangle are compared against the measured one and only
one of the six arrangements fits. On the sample recordings the correct arrangement matches to within 0.02 cm while the
next-best is 1.45 cm off. A recorded triangle that does not match the measured one, or a triangle too regular to be
identified, is reported in the warning banner.

The shipped `config/extrinsic.toml` is a placeholder with the right side lengths but a guessed pose, so the absolute
deviation still contains a systematic component until the real measurements are entered. The "consistency of deviations"
analysis quantifies how much of the deviation is frame misalignment: on the sample recordings the affine model leaves a
residual of a few millimeters while the rigid and similarity models do not, which indicates that the two systems use
opposite handedness.

The `+x` right, `+y` up, `+z` forward convention used for the markers is left-handed, while Optitrack's global frame is
right-handed, so a pure rotation can never fit both the markers and the LED at once — the marker triangle alone fits
either way, but everything off that plane (the LED) ends up mirrored. The "Allow reflection" checkbox in the extrinsic
editor (`allow_reflection` in `config/extrinsic.toml`) lets the fit pick the mirrored solution instead of a pure
rotation; it is enabled by default in the shipped configuration and reduces the mean rigid-model deviation on the
sample recordings from over two meters to about 2 cm.

## Review

Tracking data can be viewed in a player with play/pause controls, playback speed settings, and a timeline. Several
recordings may be viewed at once. It is also possible to view the entire trajectory at once.

## Analysis

The software can analyze the deviations between two concurrent recordings. One recording serves as the ground truth. The
other is analyzed in terms of:

- Mean and standard deviation of the difference from the ground truth along each axis
- Mean and standard deviation of the Euclidean distance
- Consistency of deviations: Finds the best single projective transformation to apply to all datapoints of the second
  recording to minimize deviation and calculate how that affects deviation.

## Filtering

A filter may be applied to the recordings to smooth out noise and triangulation errors. It works by fitting a smoothing
spline to the recording data. Three smoothing techniques are available, selectable in the filter panel:

- **FITPACK (per-axis)** fits x, y and z independently, penalizing jumps in the derivative at each knot
  (`UnivariateSpline`, the same as SciPy's `splrep`). This is the default and the original implementation.
- **GCV smoothing spline** fits x, y and z jointly as a single cubic spline, penalizing the integral of the squared
  second derivative (`make_smoothing_spline`). Its smoothing parameter can either be set manually or, with "Automatic
  (GCV)" checked, chosen automatically via generalized cross-validation.
- **Parametric curve (joint xyz)** also fits x, y and z jointly, sharing one knot vector across axes and penalizing
  derivative jumps like the FITPACK method, but as a single parametric curve over time rather than three independent
  functions (`make_splprep`).

Each spline may be parameterized manually through the user interface or by minimizing mean deviation from a given
ground truth recording. The result may be displayed along with the original data in the player.

## Predictive filtering

Offline smoothing may look at the whole recording; a real-time client cannot. The Oak-D takes about 25 ms from shutter
until a position exists (measured per sample from `detection_time`, median 24.8 ms but with a tail to 30.6 ms), and a
display adds more, so a reasonable motion-to-photon budget is 50–100 ms. The prediction panel replays a recording as if
it arrived in real time and asks each candidate filter for the position it cannot yet observe.

The hard part is not accuracy. Extrapolating amplifies noise in the velocity estimate by the horizon, so a predictor can
improve mean deviation while making the output stream far noisier than the motion it tracks — on `slow2`, naive
two-sample extrapolation cuts RMSE from 1.76 cm to 1.29 cm while making the output **11× jerkier**. Deviation statistics
cannot see this, so predictions are also scored on smoothness relative to the offline fit:

- **Jerk ratio** — RMS third difference against the reference. `1.0` moves exactly as much as the real motion does,
  above that is invented motion, below it is real motion being smoothed away.
- **Step ratio** — 99th percentile per-frame movement, the most directly perceptual of the three.
- **High-frequency ratio** — output power above 8 Hz, where real hand motion has little to say.

Predictors are causal by construction and the replay harness enforces it: each one only ever sees samples that had
arrived by the time it was asked. Available variants are a zero-order hold (the cost of not predicting), a
constant-velocity Kalman filter, and a speed-scheduled variant whose process noise follows a heavily smoothed speed
estimate — no single fixed setting suits both regimes, since jitter is worst when the target is nearly still and lag is
worst when it moves. Either can additionally damp only the extrapolated part of the output, which buys smoothness more
cheaply than filtering the position would, and fades towards a hold when samples go stale.

Parameters are fitted against the worst recording rather than the average, because averaging lets whichever regime is
over-represented in the tuning set pick a value that then fails elsewhere. On four held-out recordings every variant
beats not predicting at all on both accuracy and smoothness; the fixed filter gives the lowest worst-case cost while
the scheduled one is more accurate but moves more per frame.

Note that a cubic smoothing spline is the two-sided form of the same constant-velocity model, so the offline `gcv`
filter and the Kalman filter here are the same estimator with and without access to the future.