
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


class RobotEvent(object):
    pass

class mouseMove(RobotEvent):
    def __init__(self, x, y, smooth=False):
        # TODO: validate key
        self.x = x
        self.y = y
        self.smooth = smooth
    def compile(self):
        return 'mouseMove', self.x, self.y, self.smooth

    def __repr__(self):
        return 'mouseMove<x={} y={} smooth={}>'.format(self.x, self.y, self.smooth)

    def __str__(self):
        return repr(self)

class mouseClick(RobotEvent):
    def __init__(self, button="left", double=False):
        # TODO: validate key
        self.button = button
        self.double = double
    def compile(self):
        return 'mouseClick', self.button, self.double

    def __repr__(self):
        return 'mouseClick<button={} double={}>'.format(self.button, self.double)

    def __str__(self):
        return repr(self)

class mouseDrag(RobotEvent):
    def __init__(self, x, y):
        # TODO: validate key
        self.x = x
        self.y = y
    def compile(self):
        return 'mouseDrag', self.x, self.y

    def __repr__(self):
        return 'mouseDrag<x={} y={}>'.format(self.x, self.y)

    def __str__(self):
        return repr(self)

class mouseScroll(RobotEvent):
    def __init__(self, value, direction="down"):
        # TODO: validate key
        self.value = value
        self.direction = direction
    def compile(self):
        return 'mouseScroll', self.value, self.direction

    def __repr__(self):
        return 'mouseScroll<value={} direction={}>'.format(self.value, self.direction)

    def __str__(self):
        return repr(self)

class mouseToggle(RobotEvent):
    def __init__(self, button="left", down=True):
        self.button = button
        self.down = down
    
    def compile(self):
        return 'mouseToggle', self.button, self.down

    def __repr__(self):
        return 'mouseToggle<button={} down={}>'.format(self.button, self.down)

    def __str__(self):
        return repr(self)
        

