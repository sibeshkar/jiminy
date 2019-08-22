import string

from jiminy.gym import Space
from jiminy.gym.spaces import prng

from jiminy.vncdriver import constants
from jiminy.spaces import robot_event
import random

##

# New Robot Action Space:
"""
Reference:

[http://robotjs.io/docs/syntax](http://robotjs.io/docs/syntax)

[https://github.com/go-vgo/robotgo/blob/master/docs/doc.md](https://github.com/go-vgo/robotgo/blob/master/docs/doc.md)

Mouse Actions

- mouseMove(x, y, [smooth(bool)])
    - **Desc** : Move the mouse to x, y instantly or smoothly.
    - x : x-coordinate
    - y: y-coordinate
    - smooth: either move smoothly or not
- mouseClick([button], [double]) -
    - **Desc** : Clicks the mouse.
    - **button** : left, middle, right button
    - **double** : whether to double click
- mouseDrag(x, y)
    - **Desc** : Moves mouse to x, y instantly, with the mouse button held down.
    - **x** : x-coordinate
    - **y** : y-coordinate
- mouseScroll(value, [direction])
    - **Desc** : Scrolls the mouse up or down.
    - **value** : how many pixels to scroll
    - **direction** : "up" or "down"
- mouseToggle([down], [button])
    - **Desc** : Toggles mouse button.
    - **down**  ****: Accepts down or up. (default=down)
    - **button** : Accepts left, right, or middle. (default=left)

Key Actions

- keyTap(key, [modifier])
    - **Desc** : Press a single key. along with possible modified(ctrl, alt, shift)
    - **key** : which key to press
    - **modifier** : String or an array. Accepts alt, command (win), control, and shift.
- keyToggle(key, down, [modifier])
    - **Desc** : Hold down or release a key.
    - **key** : which key to press
    - **down** : Accepts down or up.
    - **modifier** : String or an array. Accepts alt, command (mac), control, and shift.
- keyTypeString(string, cpm)
    - **Desc** : Types string at cpm characters per minute
    - **string** : strings per minute
    - **cpm** : characters per minute

"""

class RobotActionSpace(Space):
    """The space of Robot actions.

    Reference : http://robotjs.io/docs/syntax

    """
    def __init__(self, screen_shape=(1024, 728)):
        self.screen_shape = screen_shape


    def contains(self,action):
        if not isinstance(action, list):
            return False
        for a in action:
            if isinstance(a, robot_event.mouseClick):
                return True
            elif isinstance(a, robot_event.mouseDrag):
                return True
            elif isinstance(a, robot_event.mouseMove):
                return True
            elif isinstance(a, robot_event.mouseScroll):
                return True
            elif isinstance(a, robot_event.mouseToggle):
                return True
            else:
                return False
            
    
    def sample(self):
        choice = random.randint(0,4)

        x = random.randint(0,self.screen_shape[0])
        y = random.randint(0,self.screen_shape[1])

        if choice == 0:
            event = robot_event.mouseClick()
        elif choice == 1:
            event = robot_event.mouseDrag(x, y)
        elif choice == 2:
            event = robot_event.mouseMove(x, y)
        elif choice == 3:
            event = robot_event.mouseScroll(random.randint(0,50))
        elif choice == 4:
            event = robot_event.mouseToggle()

        return [event]

class RobotObservationSpace(Space):
    # For now, we leave the VNC ObservationSpace wide open, since
    # there isn't much use-case for this object.
    def contains(self, x):
        return True
        