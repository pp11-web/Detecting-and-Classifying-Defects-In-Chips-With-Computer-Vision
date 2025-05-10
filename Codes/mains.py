import streamlit as st
import subprocess

st.title("298 Final Project")
st.write("Select an option to run the corresponding script.")

# Function to run a script
def run_script(script_name):
    result = subprocess.run(['python', script_name], capture_output=True, text=True)
    st.text_area(f"Output of {script_name}:", result.stdout + result.stderr, height=150)

# Buttons for running different scripts
if st.button("Data Transformation & Info"):
    run_script("Data_info.py")

if st.button("PCB Images"):
    run_script("M1.py")

if st.button("Wafer Data"):
    run_script("M2.py")
