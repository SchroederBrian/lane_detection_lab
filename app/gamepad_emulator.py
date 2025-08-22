
import vgamepad as vg

class GamepadEmulator:
    def __init__(self, max_deg=45.0):
        self.gamepad = vg.VX360Gamepad()
        self.active = False
        self.max_deg = max_deg

    def toggle(self):
        self.active = not self.active
        if not self.active:
            self.gamepad.reset()

    def update_steering(self, steering_deg):
        if self.active:
            x_value = steering_deg / self.max_deg  # -1 to 1
            self.gamepad.left_joystick_float(x_value_float=x_value, y_value_float=0.0)
            self.gamepad.update()

