import os
import emerge as em
import numpy as np
from emerge.plot import plot_sp, smith, plot_ff_polar, plot_ff
import matplotlib.pyplot as plt

HEADLESS=True
OUT = "output_booster-revamped"
os.makedirs(OUT, exist_ok=True)

"""
Capacitvely Coupled Elements
"""

# --- UNIT DEFINITIONS
mm = 0.001

# --- FREQUENCY ELEMENTS
f0 = 915e6
bw = 300e6
# f1 = f0 - bw/2
# f2 = f0 + bw/2
f1 = 600e6
f2 = 1000e6
FREQ_STEPS = 51

# --- BOARD ELEMENT
board_th = 1.6 * mm
cu_th = 0.0348 * mm
gnd_l = 85 * mm # note resonant at 78; try others; 
gnd_w = 30 * mm
pad_gap = 1 * mm

pad_w = 6 * mm
pad_l = 6 * mm

diel_l = gnd_l + pad_gap + pad_l + 0.5 * mm

er = 4.6

# --- SIMULATION PREAMBLE
model = em.Simulation("CCE_915M")
model.settings.check_ram = False
model.check_version("2.4.3")

# --- DEFINE GEOMETRY
substrate = em.geo.Box(gnd_w, 
                       diel_l, 
                       board_th, 
                       position=(0,0,-board_th)
                       ).set_material(em.Material(er, color="#207020", opacity=0.6))

pad = em.geo.Box(pad_w, 
                 pad_l, 
                 cu_th,
                 position = (0,gnd_l + pad_gap,0)
                 ).set_material(em.lib.COPPER)

gnd_top = em.geo.Box(gnd_w, 
                     gnd_l, 
                     cu_th,
                     position=(0,0,0)
                     ).set_material(em.lib.COPPER)

# gnd_bottom = em.geo.Box(gnd_w, 
#                         gnd_l, 
#                         cu_th, 
#                         position=(0,0,-board_th - cu_th)
#                         ).set_material(em.lib.COPPER)

air = em.geo.open_region(15 * mm, 15 *mm, 15 * mm).background()

port = em.geo.Plate(
        np.array([0, gnd_l, cu_th]),   # start at top edge of ground
        np.array([0.5 * mm, 0, 0]),          # width across x
        np.array([0, pad_gap, 0])         # across the gap toward the pad
        )

model.view()

# --- REFINE MESH
model.mw.set_resolution(0.3)
model.mw.set_frequency_range(f1, f2, FREQ_STEPS)

model.commit_geometry()

model.mesher.set_face_size(port, 0.3 * mm)
model.mesher.set_boundary_size(pad.face('-z'), 0.3*mm)
model.generate_mesh()
model.view(selections=[port], screenshot=f"{OUT}/mesh_initial.png")

# --- PORT DEFINITIONS
port_bc = model.mw.bc.LumpedPort(
        port, 1,
        width=0.5 * mm, 
        height=pad_gap,
        direction=em.YAX, Z0=50
        )

# --- AIRBOX SETUP
boundary_selection = air.boundary()
abc = model.mw.bc.AbsorbingBoundary(boundary_selection)

model.adaptive_mesh_refinement(frequency=f0, max_steps=2)

# --- VIEW MODEL PRIOR TO SOLVE
model.view(plot_mesh = True, 
           volume_mesh=False, 
           screenshot=f"{OUT}/mesh.png", 
           off_screen=HEADLESS)

model.view(bc=True, 
           screenshot=f"{OUT}/bc.png", 
           off_screen=HEADLESS)

# --- RUN THE SOLVER
# model.mw.solveroutine.set_solver(em.EMSolver.CUDSS)

data = model.mw.run_sweep()

freqs = data.scalar.grid.freq
freq_dense = np.linspace(f1, f2, 1001)

s11 = data.scalar.grid.model_S(1, 1, freq_dense)

plot_sp(freq_dense, s11)
plt.savefig(f"{OUT}/return_loss.png")
plt.close()

smith(s11, f=freq_dense, labels="s11")
plt.savefig(f"{OUT}/smith_s11.png")
plt.close()

# --- FAR FIELD MEASUREMENTS
ff1 = data.field.find(freq = f0).farfield_2d((0,0,1), (1,0,0), boundary_selection)
ff2 = data.field.find(freq = f0).farfield_2d((0,0,1), (0,1,0), boundary_selection)
plot_ff_polar(ff1.ang, [ff1.gain.norm, ff2.gain.norm], dB=True, dBfloor=-20)
plt.savefig(f"{OUT}/ff_polar.png")
plt.close()

# --- 3D RADIATION
model.display.populate()
field = data.field.find(freq = f0)
ff3d = field.farfield_3d(boundary_selection, origin=(0,0,0)) 
surf = ff3d.surfplot('normE', rmax=40 * mm, offset=(0, 0, 0))
model.display.add_surf(*surf.xyzf)
model.display.show(screenshot=f"{OUT}/ff_3d.png")

# --- CURRENT DISTRIBUTION
model.display.populate()

# Plot normH on each copper surface separately
model.display.add_field(field.boundary(gnd_top.face('-z')).scalar('normH', 'abs'))
model.display.add_field(field.boundary(pad.face('-z')).scalar('normH', 'abs'))

model.display.show(
    screenshot=f"{OUT}/current_distribution.png",
    off_screen=HEADLESS
)

# --- EFFICIENCY & REALIZED GAIN
# Use farfield_2d — it correctly normalises by accepted port power
# farfield_3d does NOT set Ptot automatically, causing the nan

s11_f0 = data.scalar.find(freq=f0).S(1, 1)
mismatch_eff = 1 - np.abs(s11_f0)**2

# Two principal planes (E-plane and H-plane)
ff_e = field.farfield_2d((0, 0, 1), (1, 0, 0), boundary_selection)  # E-plane (XZ)
ff_h = field.farfield_2d((0, 0, 1), (0, 1, 0), boundary_selection)  # H-plane (YZ)

# Peak gain from whichever plane has the higher value
peak_gain_e = np.max(np.abs(ff_e.gain.norm))
peak_gain_h = np.max(np.abs(ff_h.gain.norm))
peak_gain   = max(peak_gain_e, peak_gain_h)

peak_dir_e  = np.max(np.abs(ff_e.dir.norm))
peak_dir_h  = np.max(np.abs(ff_h.dir.norm))
peak_directivity = max(peak_dir_e, peak_dir_h)

radiation_eff    = peak_gain / peak_directivity
total_eff        = radiation_eff * mismatch_eff
peak_realized_gain = peak_gain * mismatch_eff

def to_dB(x): return 10 * np.log10(np.maximum(x, 1e-12))

summary = (
    f"=== Antenna Performance @ {f0/1e6:.0f} MHz ===\n"
    f"S11:                  {20*np.log10(np.abs(s11_f0)):.2f} dB\n"
    f"Mismatch efficiency:  {to_dB(mismatch_eff):.2f} dB  ({100*mismatch_eff:.1f}%)\n"
    f"Radiation efficiency: {to_dB(radiation_eff):.2f} dB  ({100*radiation_eff:.1f}%)\n"
    f"Total efficiency:     {to_dB(total_eff):.2f} dB  ({100*total_eff:.1f}%)\n"
    f"Peak directivity:     {to_dB(peak_directivity):.2f} dBi\n"
    f"Peak gain:            {to_dB(peak_gain):.2f} dBi\n"
    f"Peak realized gain:   {to_dB(peak_realized_gain):.2f} dBi\n"
)

data.scalar.grid.export_touchstone(
    f"{OUT}/antenna.s1p",
    Z0ref=50,
    format="RI",
    custom_comments=[
        "CCE 915MHz antenna simulation",
        f"Board: {gnd_l*1000:.0f}mm x {gnd_w*1000:.0f}mm",
        f"Pad: {pad_l*1000:.0f}mm x {pad_w*1000:.0f}mm, gap={pad_gap*1000:.1f}mm",
    ],
    # funit="GHz"
)
