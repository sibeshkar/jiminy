from jiminy.gym import Space
# from jiminy.wrappers.experimental import JiminyActionWrapper
from jiminy import actions
from jiminy.vectorized import Env
import numpy as np

class ClickActionSpace(Space):
    """
    Implementes the ClickActionSpace

    each action internally is doing something semantic
    and therefore wraps around JiminyActionWrapper
    """
    def __init__(self, clickType='basic'):
        """
        :param clickType: basic is for simple click actions
        """
        self.possibleActionList = []
        if clickType == 'basic':
            self.possibleActionList = [
                    actions.ClickTarget,
                    actions.DragTarget,
                    actions.HoverTarget
                    ]

    def contains(self, action):
        if not isinstance(action, list):
            return False

        for a in action:
            if (not isinstance(a, actions.ClickAction)):
                return False
            if not np.array([isinstance(a, act) for act in self.possibleActionList], dtype=np.bool).any():
                return False

        return True

    def sample(self):
        size = len(self.possibleActionList)
        return self.possibleActionList[np.random.randint(low=0, high=size)]

if __name__ == "__main__":
    cas = ClickActionSpace(clickType='basic')
    print(cas.contains([actions.ClickTarget(env=Env(), targetObject=[])]))
