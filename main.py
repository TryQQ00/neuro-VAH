from gui import main_gui

def main():
    import os
    os.makedirs('results', exist_ok=True)
    main_gui()

if __name__ == '__main__':
    main()