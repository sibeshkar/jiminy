import string

from jiminy.gym import Space
from jiminy.gym.spaces import prng

from jiminy.vncdriver import constants
from jiminy.spaces import vnc_event

class RobotActionSpace(Space):
    """The space of Robot actions.

    Reference : http://robotjs.io/docs/syntax

    Actions:

    ## Mouse Actions

    Move(x, y, [smooth(bool)])

    Click([button], [double])

    Drag(x, y)

    Scroll(x, y)

    Toggle()([down], [button])

    ## Key Actions

    Tap(key, [modifier])

    Toggle(key, down, [modifier])

    Type(string, cpm)

    """