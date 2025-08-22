
import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.gamepad_emulator import GamepadEmulator

class TestGamepadEmulator(unittest.TestCase):
    def setUp(self):
        self.emulator = GamepadEmulator(max_deg=45.0)

    def test_toggle(self):
        self.assertFalse(self.emulator.active)
        self.emulator.toggle()
        self.assertTrue(self.emulator.active)
        self.emulator.toggle()
        self.assertFalse(self.emulator.active)

    def test_update_steering(self):
        self.emulator.active = True
        self.emulator.update_steering(22.5)
        # No assertion possible without checking gamepad state, but ensures no error

if __name__ == '__main__':
    unittest.main()
