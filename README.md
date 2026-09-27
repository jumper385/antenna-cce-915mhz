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

No arguments are required. Configuration is read from `.env` when present. By default, the script opens the geometry viewer and stops before solving (`INSPECT_GEO_ONLY=True`). Set `INSPECT_GEO_ONLY=False` to run the full simulation.

With the default pad dimensions, simulation results are written to `output_gap1mm_pad6x8mm/`.

---

## Configuration

Copy [.env.example](.env.example) to `.env` and adjust values as needed. All pad dimensions are in millimetres.

| Variable | Default | Description |
|---|---:|---|
| `HEADLESS` | `False` | Hide interactive viewer windows |
| `INSPECT_GEO_ONLY` | `True` | Exit after opening the geometry, before meshing and solving |
| `PAD_GAP` | `1` | Gap between the ground plane and coupling pad |
| `PAD_W` | `6` | Coupling pad width |
| `PAD_L` | `8` | Coupling pad length |
| `OUT_DIR` | Generated | Optional output-directory override |

When `OUT_DIR` is unset, the output folder is generated from the pad parameters. For example, `PAD_GAP=1.5`, `PAD_W=6`, and `PAD_L=8` produce `output_gap1p5mm_pad6x8mm/`.

---

## Outputs

All files are saved to the selected output directory.

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

