from jiminy import vectorized
from jiminy.gym import Wrapper
from enum import Enum

class JiminyActionState(Enum):
    """
    Unlike gym like control, there is a time delay while an action is taking place
    In order to streamline we also maintain a feedback from the action
    This feedback hence tells us if the action has started, initiated, processed or finished
    """
    Initiated = 0
    StartingAction = 1
    ProcessingAction = 2
    FinishedAction = 3

class JiminyActionWrapper(vectorized.Wrapper, Wrapper):
    """
    Any Jiminy Action must extend this class to have the proper
    implementation requirements

    Any JiminyAction also returns a gradient. This gradient can be used to optimize any model which is trying to learn if performing this action is possible on the said object it was created, or the said targetObject it was provided.

    When we learn by imitation learning we return a gradient which is the log (empirical probability) this action is taken (or a normalized version of the same).

    When we learn by exploration we get some intrinsically calculated reward for the said action, and return the gradient of the wrt the action taken.
    """
    def __init__(self, env):
        super(JiminyActionWrapper, self).__init__(env)
        self.env = env
        self.state = JiminyActionState.Initiated

    def _control_algorithm(self, targetObject):
        raise NotImplementedError

    def step(self, targetObject):
        """
        Unlike the step function in gym
        we give an additional target to the step function
        this is because our actions here can perform different things and are actually meta actions

        such as clickable type might move and hover, or left-click or right-click. So instead of three different actions, having a target object ensures we can merge them into one. Even though internally all three will happen independently

        or the text which has to be typed into a text-box. While we can type it character by character and re-infer the state and understand the goal, its possible we know the entire text we want to type in the same go, in which case we would like to use this kind of a model to perform a said action
        """
        self._check(targetObject)
        self._control_algorithm(targetObject)

    def _check(self, targetObject):
        """
        Checks if the targetObject is valid for the
        action being taken by the object
        """
        raise NotImplementedError
