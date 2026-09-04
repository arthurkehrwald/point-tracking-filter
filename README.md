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

The transform from the Oak-D marker triangle to the camera's optical center is not fitted from the data. It is
hand-specified in `config/extrinsic.toml` and editable in the analysis panel. With the default identity extrinsic the
absolute deviation therefore contains a large systematic component; the "consistency of deviations" analysis quantifies
how much of it is frame misalignment. On the sample recordings the affine model leaves a residual of a few millimeters,
while the rigid and similarity models do not, which indicates that the two systems use opposite handedness.

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