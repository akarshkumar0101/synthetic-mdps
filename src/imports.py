# %load_ext autoreload
# %autoreload 2

import_str = """
import os
import sys
import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce, repeat
import gymnax
import optax
import flax
import pickle
import json

import run
import run_bc
"""

exec(import_str)
