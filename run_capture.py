import numpy as np, os
from setup_autolabel import learn_palette, auto_label, ReviewTool

frames = np.load("pilot_frames.npy")

EXCLUDE = [(0.190, 0.236, 0.270, 0.316),      # B
           (0.520, 0.735, 0.590, 0.805)]      # A

PALETTE = learn_palette(frames, exclude=EXCLUDE, area=(12, 200))

if os.path.exists("pilot_labels.json"):
    os.remove("pilot_labels.json")

auto_label(frames, "pilot_labels.json", exclude=EXCLUDE,
           palette_hues=PALETTE, area=(12, 200))

ReviewTool(frames, "pilot_labels.json").run()