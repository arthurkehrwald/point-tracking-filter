# Point tracking filter

This project is about processing 3D point tracking data of an infrared LED recorded using a camera system. The data is
loaded from CSV files.

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