from jiminy.wrappers.experimental import JiminyActionWrapper, JiminyActionState
from jiminy.vectorized import Env
from jiminy.spaces import vnc_event

class ClickAction(JiminyActionWrapper):
    pass

class ClickTarget(ClickAction):
    def _check(self, targetParams):
        assert isinstance(targetParams, dict), "Expected targetParams to be of type dict, got {}".format(targetParams)
        assert "x_1" in  targetParams, "No co-ordinate x_1 found in targetParams {}".format(targetParams)
        assert "x_2" in  targetParams, "No co-ordinate x_2 found in targetParams {}".format(targetParams)
        assert "y_1" in  targetParams, "No co-ordinate y_1 found in targetParams {}".format(targetParams)
        assert "y_2" in  targetParams, "No co-ordinate y_2 found in targetParams {}".format(targetParams)

    def _get_default_params(self):
        return self.targetObject.boundingBox

    def _control_algorithm(self, targetParams):
        target_x = (targetParams["x_1"] + targetParams["x_2"]) / 2.
        target_y = (targetParams["y_1"] + targetParams["y_2"]) / 2.
        self.state = JiminyActionState.StartingAction
        self.env.step([vnc_event.PointerEvent(target_x, target_y, 1)])
        self.env.step([vnc_event.PointerEvent(target_x, target_y, 0)])

class HoverTarget(ClickAction):
    def _check(self, targetParams):
        assert isinstance(targetParams, dict), "Expected targetParams to be of type dict, got {}".format(targetParams)
        assert "x_1" in  targetParams, "No co-ordinate x_1 found in targetParams {}".format(targetParams)
        assert "x_2" in  targetParams, "No co-ordinate x_2 found in targetParams {}".format(targetParams)
        assert "y_1" in  targetParams, "No co-ordinate y_1 found in targetParams {}".format(targetParams)
        assert "y_2" in  targetParams, "No co-ordinate y_2 found in targetParams {}".format(targetParams)

    def _get_default_params(self):
        return self.targetObject.boundingBox

    def _control_algorithm(self, targetParams):
        target_x = (targetParams["x_1"] + targetParams["x_2"]) / 2.
        target_y = (targetParams["y_1"] + targetParams["y_2"]) / 2.
        self.state = JiminyActionState.StartingAction
        self.env.step([
            vnc_event.PointerEvent(target_x, target_y, 0)
            ])


class DragTarget(ClickAction):
    def _check_dict(self, targetParams):
        assert isinstance(targetParams, dict), "Expected targetParams to be of type dict, got {}".format(targetParams)
        assert "x_1" in  targetParams, "No co-ordinate x_1 found in targetParams {}".format(targetParams)
        assert "x_2" in  targetParams, "No co-ordinate x_2 found in targetParams {}".format(targetParams)
        assert "y_1" in  targetParams, "No co-ordinate y_1 found in targetParams {}".format(targetParams)
        assert "y_2" in  targetParams, "No co-ordinate y_2 found in targetParams {}".format(targetParams)

    def _check(self, targetParams):
        assert isinstance(targetParams, tuple), "Expected targetParams to be of type tuple, got {}".format(targetParams)
        self._check_dict(targetParams[0])
        self._check_dict(targetParams[1])

    def _get_default_params(self):
        return (self.targetObject.boundingBox, self.targetObject.boundingBox)

    def _control_algorithm(self, targetParams):
        source_x = (sourceParams[0]["x_1"] + sourceParams[0]["x_2"]) / 2.
        source_y = (sourceParams[0]["y_1"] + sourceParams[0]["y_2"]) / 2.
        target_x = (targetParams[1]["x_1"] + targetParams[1]["x_2"]) / 2.
        target_y = (targetParams[1]["y_1"] + targetParams[1]["y_2"]) / 2.
        self.state = JiminyActionState.StartingAction
        self.env.step([vnc_event.PointerEvent(source_x, source_y, 1)])
        self.env.step([vnc_event.PointerEvent(target_x, target_y, 1)])
        self.env.step([vnc_event.PointerEvent(target_x, target_y, 0)])

if __name__ == "__main__":
    from jiminy.envs import DummyVNCEnv
    env = DummyVNCEnv()
    env.configure(_n=1)
    clicker = HoverTarget(env=env, targetObject=[])
    clicker._control_algorithm({
        "x_1" : 10,
        "y_1" : 10,
        "x_2" : 30,
        "y_2" : 30
        })

