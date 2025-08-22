#!/usr/bin/env python3
import sys
from PySide6 import QtWidgets
from app.main_window import MainWindow
import logging

def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        app = QtWidgets.QApplication(sys.argv)
        win = MainWindow()
        # After win creation, config is loaded, so set logging level
        level = getattr(logging, win.config.logging_level.upper(), logging.INFO)
        logging.getLogger().setLevel(level)
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
    except Exception as e:
        logging.exception("Unhandled exception in main")
        sys.exit(1)

if __name__ == "__main__":
    main()


