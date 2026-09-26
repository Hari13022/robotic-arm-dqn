import streamlit as st
import numpy as np
import torch
import pybullet as p
import pybullet_data

st.title("DQN-IK Test Dashboard 🚀")

st.write("✅ Streamlit and PyBullet are working correctly!")

if st.button("Run PyBullet Simulation"):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf")
    cube = p.loadURDF("r2d2.urdf", [0, 0, 1])
    st.write("PyBullet simulation started! Close the window to continue.")
    st.stop()
