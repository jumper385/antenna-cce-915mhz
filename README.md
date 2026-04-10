# 915 MHz Booster Antenna Simulation Guide (Snippet-First)

This is a build-it-yourself tutorial for simulating a 915 MHz PCB booster antenna with EMerge.

Goal:
- Build geometry from scratch
- Define a lumped port feed correctly
- Run a sweep and inspect S11 and Zin
- Add a 2-element lumped matching network in post-processing

## 1) Start With Imports and Global Knobs

**Significance:** Top-level knobs and imports set up the entire workflow. This is your one place to change simulation parameters without diving into code.

**Key parameters:**
- `mm = 0.001`: unit conversion (1 mm = 0.001 m); use `* mm` to keep geometry readable
- `f0 = 915e6`: target design frequency; all matching happens at this frequency
- `f1`, `f2`: sweep lower and upper bounds; wider sweep = longer runtime but better bandwidth insight
- `Z0 = 50`: S-parameter reference impedance (usually 50 ohm for RF)
- `USE_SECOND_PORT`: toggle if you want to analyze dual-pad behavior
- `check_version()`: ensures you're on a compatible EMerge release

Copy this first. Keep all tunable values near the top.

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff

OUT = Path("output_booster_915M")
OUT.mkdir(exist_ok=True)

mm = 0.001
f0 = 915e6
f1 = f0 - 200e6
f2 = f0 + 200e6
Z0 = 50
USE_SECOND_PORT = False

model = em.Simulation("Booster_Ant_915")
model.check_version("2.4.3")
```

## 2) Define Materials and Board Dimensions

**Significance:** These parameters define the electromagnetic environment. Dielectric permittivity changes resonance; board dimensions directly affect radiation and matching.

**Key parameters:**
- `fr4_er = 4.15`: FR4 relative permittivity; higher ε slows wavelength and lowers resonance
- `fr4_tand = 0.013`: FR4 loss tangent; controls dissipation; typical values 0.01–0.015
- `pcb_h`: substrate height; thicker boards reduce coupling and shift resonance downward
- `copper_t`: thickness of copper traces; typically 0.035 mm (1 oz) or 0.070 mm (2 oz)
- `gnd_W`, `gnd_L`: ground plane size; larger ground increases directivity and lowers resonance
- `pad_s`: coupling pad size; larger pad increases capacitive coupling, lowering resonance
- `gap`: slot width between ground and pad; smaller gap = stronger coupling and higher impedance

```python
fr4_er   = 4.15
fr4_tand = 0.013
pcb_h    = 1.0   * mm
copper_t = 0.035 * mm

gnd_W = 120 * mm
gnd_L = 50  * mm

pad_s = 6   * mm
gap   = 0.5 * mm
```

## 3) Build Geometry (Substrate, Ground, Coupling Pads)

**Significance:** Geometry defines the electromagnetic structure. The substrate sets permittivity; ground provides reference current path; pads create the feed coupling. Spatial accuracy is critical—misaligned geometry causes simulation errors.

**Key concepts:**
- **Substrate**: dielectric volume that sets relative permittivity; must be large enough to contain near-field
- **Ground plane**: main current-return path; defines radiation reference and board geometry
- **Coupling pads**: small conductors that couple RF into/out of the antenna via capacitive gap
- **Position tuples** `(x, y, z)`: indicate where geometry is placed in 3D space; origin is at (0,0,0) by default

```python
substrate = em.geo.Box(
	gnd_W + gap + pad_s, gnd_L, pcb_h,
	position=(0, 0, 0)
).set_material(em.Material(fr4_er, tand=fr4_tand, color="#207020", opacity=0.6))

ground = em.geo.Box(
	gnd_W, gnd_L, copper_t,
	position=(0, 0, pcb_h)
).set_material(em.lib.COPPER)

z_cu = pcb_h

pad1 = em.geo.Box(
	pad_s, pad_s, copper_t,
	position=(gnd_W + gap, gnd_L - pad_s, z_cu)
).set_material(em.lib.COPPER)

pad2 = em.geo.Box(
	pad_s, pad_s, copper_t,
	position=(gnd_W + gap, 0, z_cu)
).set_material(em.lib.COPPER)
```

## 4) Define the Feed as a Lumped Port Plate

**Significance:** This is critical. The port plate defines the electromagnetic feed cross-section. Incorrect plate orientation or size causes S11 to be wrong even if geometry is valid. The solver needs to know where—and in what direction—to excite the structure.

**Key parameters:**
- `Plate(origin, width_vector, height_vector)`: defines a rectangular face in 3D space
  - `origin`: corner of the plate starting point
  - `width_vector`: direction and magnitude along one dimension of the plate
  - `height_vector`: direction and magnitude perpendicular to width
- The plate must span the physical gap completely
- Area = width magnitude × height magnitude; this is the feed cross-section

**If port plate is wrong:**
- Mismatch between plate and LumpedPort parameters → S11 reads incorrectly
- Port area too small → fields over-compress, high impedance seen
- Port area too large → fields spread, low impedance seen

This is the most important part.

The feed plate spans the physical gap where excitation happens.

```python
port1_face = em.geo.Plate(
	np.array([gnd_W,          gnd_L - pad_s, z_cu]),
	np.array([gap,            0,             0   ]),
	np.array([0,              pad_s,         0   ])
)

port2_face = em.geo.Plate(
	np.array([gnd_W, 0,     z_cu]),
	np.array([gap,   0,     0   ]),
	np.array([0,     pad_s, 0   ])
)
```

How to think about this:
- Plate width vector points across the small coupling gap in X
- Plate height vector follows pad width in Y
- Electric field direction should also be across the gap: X axis

## 5) Add Open Boundary, Mesh, and Frequency Sweep

**Significance:** Open boundary prevents artificial reflections from the simulation domain edges (absorbing boundary condition). Meshing converts continuous geometry into discrete elements; coarse mesh is fast but inaccurate; fine mesh is accurate but slow. Frequency sweep defines all frequencies to solve at.

**Key parameters:**
- `open_region(30mm, 30mm, 30mm)`: box size around antenna; larger = less edge reflection but slower; 3–5× antenna size typical
- `.background()`: makes the air box the background material
- `set_resolution(7)`: mesh element size as fraction of wavelength; 7 = ~λ/7 fine (coarse); typical 5–10; lower = finer but slower
- `set_frequency_range(f1, f2, 21)`: solve at 21 equally-spaced frequencies from f1 to f2; more points = smoother S11 curve but slower
- `set_face_size()`: locally refine mesh on specific faces (e.g., port); refine here because fields are strongest
- `set_boundary_size()`: locally refine on edges; edge current diffraction needs fine mesh
- `commit_geometry()`: lock geometry before meshing
- `generate_mesh()`: build the finite-element mesh

```python
air = em.geo.open_region(30 * mm, 30 * mm, 30 * mm).background()
boundary_selection = air.boundary()

model.mw.set_resolution(7)
model.mw.set_frequency_range(f1, f2, 21)

model.commit_geometry()
model.mesher.set_face_size(port1_face, 0.25 * mm)
model.mesher.set_boundary_size(pad1.face('+z'), 1 * mm)
if USE_SECOND_PORT:
	model.mesher.set_face_size(port2_face, 0.25 * mm)
	model.mesher.set_boundary_size(pad2.face('+z'), 1 * mm)
model.generate_mesh()
```

## 6) Assign Lumped Port Boundary Condition

**Significance:** This is where you tell the solver *how* to excite the antenna. A lumped port applies voltage across a small gap (like a 50Ω transmission line feeding a gap). Without a port, geometry alone does nothing—there's no source. The boundary condition (BC) is the interface between solver and physics.

**Key parameters:**
- `port1_face, 1`: which plate face (defined above), and port number (integer ID)
- `width`, `height`: must match the port plate dimensions; if mismatched, impedance calculation is wrong
- `direction=em.XAX`: which axis has the E field; XAX = X-axis, YAX = Y-axis, ZAX = Z-axis
- `Z0=50`: reference impedance for S-parameters *only*; does **not** force the antenna to match 50 ohm. It's the denominator in $Z_{in} = Z_0 \frac{1+S_{11}}{1-S_{11}}$
- `AbsorbingBoundary()`: on the air box boundary; prevents EM waves from bouncing back into the domain

**If port BC is wrong:**
- Mismatched width/height → S11 reads at wrong impedance scale
- Wrong direction → E field applied in wrong axis, antenna may not excite

```python
model.mw.bc.LumpedPort(
	port1_face, 1,
	width=pad_s,
	height=gap,
	direction=em.XAX,
	Z0=Z0
)

if USE_SECOND_PORT:
	model.mw.bc.LumpedPort(
		port2_face, 2,
		width=pad_s,
		height=gap,
		direction=em.XAX,
		Z0=Z0
	)

model.mw.bc.AbsorbingBoundary(boundary_selection)
```

## 7) Solve and Get S11

**Significance:** The solver runs the finite-element method (FEM) across all frequencies. Adaptive mesh refinement auto-refines at your target frequency to maximize accuracy there. Post-processing densely interpolates S11 (you get 1001 points vs the 21 frequency points solved).

**Key parameters:**
- `adaptive_mesh_refinement(frequency=f0, max_steps=4)`: iteratively refines mesh at f0 until convergence; 4 steps typical
- `run_sweep()`: solves at all f1 to f2 points; returns field and S-parameter data
- `model_S(1, 1, freq_dense)`: extracts S11 at dense frequency points; first '1' = source port, second '1' = destination (same port = reflection)
- Dense interpolation: creates smooth curves instead of rough 21-point data

**Key outputs:**
- S11: reflection coefficient; |S11| close to 1 = poor match, close to 0 = good match
- Smith chart: visual representation of impedance trajectory across frequency

```python
model.adaptive_mesh_refinement(frequency=f0, max_steps=4)
data = model.mw.run_sweep()

freq_dense = np.linspace(f1, f2, 1001)
S11 = data.scalar.grid.model_S(1, 1, freq_dense)

plot_sp(freq_dense, S11, labels='S11 (active pad)')
plt.savefig(OUT / "s11.png")
plt.close()

smith(S11, f=freq_dense, labels='S11')
plt.savefig(OUT / "smith.png")
plt.close()
```

## 8) Convert S11 to Input Impedance and VSWR

**Significance:** S11 alone doesn't tell you impedance; you need the impedance formula. Zin = Rin + j*Xin where Rin is real (resistance) and Xin is imaginary (reactance). At matched condition, Zin ≈ 50 ohm (real). Reactive part (Xin) shows how far off resonance you are. VSWR is another matching metric: 1.0 = perfect, 2.0 = acceptable, >2 = poor.

**Key parameters:**
- `1e-15` in denominator: numerical guard against divide-by-zero when S11 ≈ 1
- `Rin`: real part; should be close to Z0 for good match
- `Xin`: imaginary part; should be close to 0 at resonance (tuning indicator)
- `Gamma = |S11|`: magnitude of reflection coefficient
- VSWR = (1 + |Γ|) / (1 - |Γ|); 1.0 = matched, higher = mismatch
- `i915`: index of frequency point closest to f0; lets you extract metrics at exactly 915 MHz

Use this to diagnose matching at 915 MHz.

```python
Zin = Z0 * (1 + S11) / (1 - S11 + 1e-15)
Rin = np.real(Zin)
Xin = np.imag(Zin)

Gamma = np.abs(S11)
vswr = (1 + Gamma) / np.maximum(1 - Gamma, 1e-6)

i915 = int(np.argmin(np.abs(freq_dense - f0)))
print(
	f"915 MHz: S11={20*np.log10(np.abs(S11[i915]) + 1e-15):.2f} dB, "
	f"Zin={Rin[i915]:.2f} + j{Xin[i915]:.2f} ohm, VSWR={vswr[i915]:.2f}"
)
```

Math used:

$$
Z_{in} = Z_0 \frac{1+S_{11}}{1-S_{11}}
$$

## 9) Add a 2-Element Lumped Matching Network (Post-Processing)

**Significance:** Raw Zin is likely not 50 ohm. Rather than re-simulate with different geometry, we compute the best L/C network in post-processing. This is fast and lets you instantly see what matching would look like with real components. Two topologies (series-then-shunt vs shunt-then-series) are tried; best one by error metric wins.

**Key concepts:**
- Series element: adds reactance in series with load (ZL + jX)
- Shunt element: adds admittance in parallel with load (1/YL + Y)
- Topology order matters: which comes first affects final impedance
- Search space: practical values for hand-solderable parts (0.1 nH–100 nH inductors, 0.1 pF–100 pF capacitors)
- Best candidate: minimizes |Zin_matched - 50| at f0

This section automatically picks a series + shunt L/C network at 915 MHz.

```python
def series_reactance_from_value(kind, value, omega):
	if kind == 'L':
		return 1j * omega * value
	return -1j / (omega * value)

def shunt_admittance_from_value(kind, value, omega):
	if kind == 'C':
		return 1j * omega * value
	return -1j / (omega * value)

def apply_match_topology(zload, omega, topology, s_kind, s_val, p_kind, p_val):
	zs = series_reactance_from_value(s_kind, s_val, omega)
	yp = shunt_admittance_from_value(p_kind, p_val, omega)
	if topology == 'series_then_shunt':
		return zs + 1 / (1 / zload + yp)
	return 1 / (yp + 1 / (zload + zs))

def find_best_match(zload_f0, omega0, Z0=50):
	series_vals = np.logspace(-10, -7, 81)
	shunt_vals  = np.logspace(-13, -10, 81)
	candidates = []

	for topology in ['series_then_shunt', 'shunt_then_series']:
		for s_kind in ['L', 'C']:
			for p_kind in ['C', 'L']:
				for s_val in series_vals:
					for p_val in shunt_vals:
						zin_try = apply_match_topology(
							zload_f0, omega0, topology,
							s_kind, s_val, p_kind, p_val
						)
						err = np.abs(zin_try - Z0)
						candidates.append((err, topology, s_kind, s_val, p_kind, p_val, zin_try))

	candidates.sort(key=lambda t: t[0])
	return candidates[0]
```

Apply it at 915 MHz and generate matched S11:

```python
omega0 = 2 * np.pi * f0
best = find_best_match(Zin[i915], omega0, Z0=Z0)
_, topo_best, s_kind_best, s_val_best, p_kind_best, p_val_best, zin_best = best

omega_dense = 2 * np.pi * freq_dense
Zin_matched = np.empty_like(Zin, dtype=complex)
for k, (zl, om) in enumerate(zip(Zin, omega_dense)):
	Zin_matched[k] = apply_match_topology(
		zl, om, topo_best,
		s_kind_best, s_val_best,
		p_kind_best, p_val_best
	)

S11_matched = (Zin_matched - Z0) / (Zin_matched + Z0)
```

Save chosen component values:

```python
with open(OUT / 'matching_network.txt', 'w', encoding='ascii') as f:
	f.write('Automatic 2-element match synthesis at 915 MHz\n')
	f.write(f'Topology: {topo_best}\n')
	f.write(f'Series: {s_kind_best} value={s_val_best}\n')
	f.write(f'Shunt:  {p_kind_best} value={p_val_best}\n')
	f.write(f'Raw Zin @ 915 MHz: {Zin[i915].real:.3f} + j{Zin[i915].imag:.3f} ohm\n')
	f.write(f'Matched Zin @ 915 MHz: {zin_best.real:.3f} + j{zin_best.imag:.3f} ohm\n')
```

## 10) Plot Raw vs Matched Performance

**Significance:** Overlay comparison shows the impact of the matching network. You see how much S11 improved and over what bandwidth. Dashed line at -10 dB is a common design target (reflection loss ~0.4 dB).

**Key parameters:**
- `20*log10(|S11|)`: converts linear reflection coefficient to dB scale
- `-10 dB` line: rule-of-thumb acceptable threshold; tighter specs might use -15 dB
- `dpi=150`: print resolution; higher = sharper but larger files

```python
vswr_matched = (1 + np.abs(S11_matched)) / np.maximum(1 - np.abs(S11_matched), 1e-6)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(freq_dense / 1e9, 20*np.log10(np.abs(S11) + 1e-15), label='S11 raw')
ax.plot(freq_dense / 1e9, 20*np.log10(np.abs(S11_matched) + 1e-15), label='S11 matched')
ax.axhline(-10, color='k', linestyle='--', linewidth=1, label='-10 dB')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('S11 [dB]')
plt.tight_layout()
plt.savefig(OUT / 's11_matched.png', dpi=150)
plt.close()
```

## 11) Optional Far-Field Snippets

**Significance:** Far-field patterns show how the antenna radiates—directivity, nulls, gain. E-plane and H-plane are two orthogonal 2D cuts through the 3D radiation pattern. Gain in dBi is referenced to isotropic radiator.

**Key parameters:**
- `find(freq=f0)`: extract field data at this frequency
- `(0, 0, 1)`: normal vector to the plane (here: Z-axis = looking down on antenna)
- `(0, 1, 0)`: second vector defining the cut direction (E-plane: Y-axis)
- `(1, 0, 0)`: alternative second vector (H-plane: X-axis)
- Pattern amplitude in dBi; negative dBi = less efficient than isotropic

```python
field_915 = data.field.find(freq=f0)
ff_eplane = field_915.farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)
ff_hplane = field_915.farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)

plot_ff(
	[ff_eplane.ang * 180 / np.pi, ff_hplane.ang * 180 / np.pi],
	[ff_eplane.gain.norm, ff_hplane.gain.norm],
	dB=True, ylabel='Gain [dBi]', labels=['E-plane', 'H-plane']
)
plt.savefig(OUT / "farfield.png")
plt.close()
```

## 10b) S-Parameter Renormalization to Z_source(f)

**Significance:** `S11_matched` from section 9 tells you the reflection at the *input* of the matching network (the 50 Ω generator side). The antenna port sees a different source impedance — the Thévenin equivalent looking *back through the matching network* toward the generator. Renormalizing S11 to that source impedance gives you what a VNA would read if you probed directly at the antenna pad after soldering the 0402 components. It also cross-checks your topology math: if both values agree at f0, the renormalization is self-consistent.

**Why not just change `Z0` on `LumpedPort`?**  
`LumpedPort.Z0` only accepts a real float (it's a sheet resistance). The source impedance here is strongly reactive (`Z_src ≈ 9 + j269 Ω` at 915 MHz) and varies with frequency. Injecting only the real part would give wrong physics. Post-processing renormalization handles the full complex, frequency-dependent Z_src exactly.

**Key parameters:**
- `Z_series_v`, `Z_shunt_v`: element impedance/admittance evaluated at every frequency in `omega_dense`
- `Z_src`: Thévenin source impedance seen at the antenna port, computed from the matching topology
- `S11_renorm = (Zin - conj(Z_src)) / (Zin + Z_src)`: standard complex renormalization; conjugate in numerator ensures power-wave definition

**Topology-specific Z_src formulae:**

For `shunt_then_series` (shunt element first from generator, then series toward antenna):
$$Z_{src}(f) = Z_{series}(f) + \frac{50 \cdot Z_{shunt}(f)}{50 + Z_{shunt}(f)}$$

For `series_then_shunt` (series element first from generator, then shunt toward antenna):
$$Z_{src}(f) = \frac{Z_{shunt}(f)\,(50 + Z_{series}(f))}{Z_{shunt}(f) + 50 + Z_{series}(f)}$$

```python
def _Zelement_vec(kind, val, omega_arr):
    if kind == 'L':
        return 1j * omega_arr * val
    return 1.0 / (1j * omega_arr * val + 1e-300)

Z_series_v = _Zelement_vec(s_kind_best, s_val_best, omega_dense)
Z_shunt_v  = _Zelement_vec(p_kind_best, p_val_best, omega_dense)

if topo_best == 'shunt_then_series':
    Z_src = Z_series_v + 50.0 * Z_shunt_v / (50.0 + Z_shunt_v)
else:
    Z_src = Z_shunt_v * (50.0 + Z_series_v) / (Z_shunt_v + 50.0 + Z_series_v)

S11_renorm = (Zin - np.conj(Z_src)) / (Zin + Z_src)
```

**Sanity check:** at f0, `S11_renorm` and `S11_matched` should agree to within numerical noise. If they differ, the topology constants are wired incorrectly.

Plot all three curves together for full picture:

```python
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(freq_dense / 1e9, 20*np.log10(np.abs(S11) + 1e-15),
        label='S11 raw (EM)')
ax.plot(freq_dense / 1e9, 20*np.log10(np.abs(S11_matched) + 1e-15),
        label=f'S11 matched (source port, {topo_best})')
ax.plot(freq_dense / 1e9, 20*np.log10(np.abs(S11_renorm) + 1e-15),
        label='S11 renorm to Z_source(f) at antenna port', linestyle='--')
ax.axhline(-10, color='k', linestyle='--', linewidth=1, label='-10 dB')
ax.axvline(f0 / 1e9, color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('Frequency (GHz)'); ax.set_ylabel('S11 [dB]')
ax.grid(True, alpha=0.3); ax.legend(); plt.tight_layout()
plt.savefig(OUT / 's11_matched.png', dpi=150)
plt.close()

smith([S11, S11_renorm], f=freq_dense, labels=['S11 raw', 'S11 renorm Z_source'])
plt.savefig(OUT / 'smith_matched.png')
plt.close()
```

Append the renorm results to the matching network report:

```python
with open(OUT / 'matching_network.txt', 'a', encoding='ascii') as fout:
    fout.write(f'\nZ_source @ 915 MHz: {Z_src_f0.real:.3f}+j{Z_src_f0.imag:.3f} ohm\n')
    fout.write(f'S11 renorm @ 915 MHz: {S11_renorm_db_f0:.3f} dB\n')
```

**Outputs:**
- `s11_matched.png`: raw EM, source-side matched, and antenna-port renorm curves overlaid
- `smith_matched.png`: Smith chart comparing raw trajectory vs renormalized trajectory
- `matching_network.txt`: appended with Z_source and renorm S11 at f0

## 12) Quick Checklist Before You Run

- Frequency range (`f1`, `f2`) includes 915 MHz
- Gap is small and positive
- Port direction matches your gap axis
- Geometry committed before meshing
- Lumped port assigned before sweep

Run:

```bash
python3 booster-915M.py
```

If S11 is still poor at 915 MHz, tune in this order:
- gap
- pad size
- pad position
- ground dimensions
- then regenerate matching network values
