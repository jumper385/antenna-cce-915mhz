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
f1 = f0 - bw/2
f2 = f0 + bw/2
FREQ_STEPS = 11

# --- BOARD ELEMENTS
board_th = 1.6 * mm
cu_th = 0.0348 * mm
gnd_l = 100 * mm
gnd_w = 30 * mm
ant_clearance = 12 * mm
pad_gap = 10 * mm

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

gnd_bottom = em.geo.Box(gnd_w, 
                        gnd_l, 
                        cu_th, 
                        position=(0,0,-board_th - cu_th)
                        ).set_material(em.lib.COPPER)

air = em.geo.open_region(15 * mm, 15 *mm, 15 * mm).background()

port = em.geo.Plate(
        np.array([0, gnd_l, cu_th]),   # start at top edge of ground
        np.array([0.5 * mm, 0, 0]),          # width across x
        np.array([0, pad_gap, 0])         # across the gap toward the pad
        )

model.view()

# --- REFINE MESH
model.mw.set_resolution(3)
model.mw.set_frequency_range(f1, f2, FREQ_STEPS)

model.commit_geometry()

model.mesher.set_face_size(port, 0.5 * mm)
model.mesher.set_boundary_size(pad.face('-z'), 0.5*mm)
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
