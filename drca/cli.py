import os

def run_gui():
    """Locates the gui.py file in the installed package and runs it via Streamlit."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gui_path = os.path.join(current_dir, 'gui.py')
    
    print("Launching DRCA Interface...")
    os.system(f"streamlit run {gui_path}")