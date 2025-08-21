#!/usr/bin/env python3
import sys
from PySide6 import QtWidgets
from app.main_window import MainWindow

def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    rc = app.exec()
    
    # Graceful shutdown
    win.config_manager.update_active_config(win.config)
    win.config_manager.save()
    if win.worker.isRunning():
        win.worker.stop()
        win.worker.wait(1000)
    if win.screenWorker.isRunning():
        win.screenWorker.stop()
        win.screenWorker.wait(1000)
        
    sys.exit(rc)

if __name__ == "__main__":
    main()


