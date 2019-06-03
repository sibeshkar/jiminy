from jiminy.vectorized import ActionWrapper, ObservationWrapper

class RepresentationWrapper(ActionWrapper, ObservationWrapper):
    """
    RepresentationWrapper extends both ActionWrapper and ObservationWrapper
    Since we can get both observations from RW
    and perform actions on it
    """
    def _step(self, action):
        """
        RepresentationWrapper defines a specific type of _step function
        that is consistent with combining ActionWrapper, ObservationWrapper as
        is required by the use case
        """
        action = self.action(action)
        observation, reward, done, info = self.env.step(action)
        return self.observation(observation),reward, done, info
