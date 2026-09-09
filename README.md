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

A filter may be applied to the recordings to smooth out noise and triangulation errors. It works by fitting a spline to
the recording data. This spline may be parameterized manually through the user interface or by minimizing mean deviation
from a given ground truth recording. The result may be displayed along with the original data in the player.

Predictive filtering