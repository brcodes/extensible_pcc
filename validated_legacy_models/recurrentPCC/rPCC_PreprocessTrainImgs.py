"""Preprocess CVCV_12 training images into CVCV_12_prepro.pkl.

Read each PNG in data/raw/CVCV_12, convert to greyscale, normalize to
[0, 1], and pickle the resulting {word_id: array} dict. The original
CVCV_all set (all 212 images) is not included here, but
CVCV_all_prepro.pkl is in data/. This is part of the validated legacy
reference implementation.
"""

from __future__ import annotations

import os
import cv2
import pickle
import glob
import re
from pathlib import Path

script_dir = Path(__file__).resolve().parent
in_dir = str(script_dir / 'data' / 'raw' / 'CVCV_12')
out_dir = str(script_dir / 'data')

os.makedirs(out_dir, exist_ok=True)

I_filenames = sorted(glob.glob(os.path.join(in_dir, "*.png")))
I_dict = {}

for i in I_filenames:
    I = cv2.imread(i)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I/255

    I_id = re.sub(r".*_(.*)\.png", r"\1", i)

    I_dict.update({I_id: I})

# Pickle the output
with open(os.path.join(out_dir, "CVCV_12_prepro.pkl"), "wb") as f:
    pickle.dump(I_dict, f)