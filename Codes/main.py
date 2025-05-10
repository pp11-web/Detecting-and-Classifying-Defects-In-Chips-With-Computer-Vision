import tkinter as tk
import subprocess

def run_m1_script():
    # Run M1.py
    subprocess.run(['python', 'M1.py'])

def run_m2_script():
    # Run M2.py
    subprocess.run(['python', 'M2.py'])

def run_m3_script():
    # Run M2.py
    subprocess.run(['python', 'Data_info.py'])

window = tk.Tk()
window.title("298 Final Project")
window.geometry("500x400")

run_button_m3 = tk.Button(window, text="Data Transformation & info", command=run_m3_script)
run_button_m3.pack(pady=10)

run_button_m1 = tk.Button(window, text="PCB Images", command=run_m1_script)
run_button_m1.pack(pady=10)

run_button_m2 = tk.Button(window, text="Wafer Data", command=run_m2_script)
run_button_m2.pack(pady=10)

window.mainloop()