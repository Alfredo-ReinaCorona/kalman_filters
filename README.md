# Various Kalman filter examples
Naming convention used: "[type_of_filter] _ [state_space_dmensions] _ [measurement dimension].py"

## Notes
- Leaving `P = np.eye()` can lead to the filter being too confident too early. It is best to change it to `P = 100 * np.eye()` as this essentially tels the filter that you are not confident in the current state. This value however is imporved as you iterate through the algorithm, though with such a simple example the benefit of this is not noticable and may actually hurt early estimation. Keep in mind that `P` is the covariance (uncertainty) matrix.

## References
- [Online Kalman Filter Tutorial](https://kalmanfilter.net/kalman-filter-tutorial.html)
