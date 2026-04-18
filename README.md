# Capacitively Coupled Antenna

A full-wave EM simulation of a capacitively coupled element (CCE) antenna tuned for 915 MHz ISM-band operation. The antenna consists of a copper ground plane and a small coupling pad separated by a 1 mm gap on an FR4 substrate. The simulation sweeps 600 MHz–1 GHz, solves for S-parameters and far-field radiation, and reports key antenna figures of merit (efficiency, gain, directivity).

---

## Dependencies

Requires Python 3 and the `emerge` EM simulation package (version ≥ 2.4.3):

```bash
pip install emerge
```

All other dependencies (NumPy, Matplotlib, PyVista, SciPy, etc.) are installed automatically as emerge dependencies. A full pinned list is in [requirements.txt](requirements.txt).

---

## Quick Start

```bash
python booster-barebones.py
```

The script runs headless by default (`HEADLESS=True`). No arguments are required. Simulation results are written to `output_booster-revamped/`.

---

## Outputs

All files are saved to `output_booster-revamped/`.

| File | Description |
|---|---|
| `mesh_initial.png` | Mesh view at the port before adaptive refinement |
| `mesh.png` | Final mesh after adaptive refinement |
| `bc.png` | Boundary condition visualisation |
| `return_loss.png` | S11 return loss vs. frequency |
| `smith_s11.png` | Smith chart of S11 |
| `ff_polar.png` | Far-field gain polar plot (E-plane and H-plane) |
| `ff_3d.png` | 3D radiation pattern surface plot |
| `current_distribution.png` | Surface current (normH) on ground plane and pad |
| `antenna.s1p` | Touchstone S1P file (RI format, 50 Ω reference) |

A performance summary is also printed to the console at the end of the run:

```
=== Antenna Performance @ 915 MHz ===
S11:                  -XX.XX dB
Mismatch efficiency:  -X.XX dB  (XX.X%)
Radiation efficiency: -X.XX dB  (XX.X%)
Total efficiency:     -X.XX dB  (XX.X%)
Peak directivity:     X.XX dBi
Peak gain:            X.XX dBi
Peak realized gain:   X.XX dBi
```

