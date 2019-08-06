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
    def __init__(self, env, targetObject):
        if env is None:
            raise ValueError
        if targetObject is None:
            raise ValueError
        super(JiminyActionWrapper, self).__init__(env)
        self.env = env
        self.targetObject = targetObject
        self.state = JiminyActionState.Initiated

    def _control_algorithm(self):
        """
        This should only be called once the targetObject and the targetParams have been set
        """
        raise NotImplementedError

    def control_algorithm(self):
        if self.targetObject is None:
            raise ValueError
        if self.targetParams is None:
            raise ValueError
        self._control_algorithm()

    def step(self):
        if self.targetParams is None:
            self.targetParams = self._get_default_params()
        return self._control_algorithm(self.targetParams)

    def setTarget(self, targetParams=None):
        if targetParams is None:
            targetParams = self._get_default_params()
        self.targetParams = targetParams
        self._check(targetParams)
        # return object for chaining
        return self

    def _check_params(self, targetParams):
        """
        Checks if the targetObject is valid for the
        action being taken by the object
        """
        raise NotImplementedError

    def _get_default_params(self):
        raise NotImplementedError
