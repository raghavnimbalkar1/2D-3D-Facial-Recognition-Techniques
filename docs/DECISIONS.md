# Decisions

- This sprint compares 2D texture with pseudo-3D monocular geometry.
- Real 3D sensors, point-cloud IO, ICP, Torch, depth CNNs, and a Streamlit app
  are deferred until licensed Tier-A data is available.
- Verification scores are similarities: larger means more likely same person.
- EER uses linear interpolation at the FAR/FRR crossing.
