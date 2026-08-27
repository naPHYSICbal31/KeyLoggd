import tkinter as tk
from tkinter import font as tkfont

root = tk.Tk()
root.withdraw()

families = tkfont.families()
print(f"Total fonts Tk sees: {len(families)}\n")

matches = [f for f in families if "roboto" in f.lower()]
if matches:
    print("Roboto-family matches:")
    for m in matches:
        print(f"  {m!r}")
else:
    print("No family containing 'roboto' found.")