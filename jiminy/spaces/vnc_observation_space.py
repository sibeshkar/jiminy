from jiminy.gym import Space

class VNCObservationSpace(Space):
    # For now, we leave the VNC ObservationSpace wide open, since
    # there isn't much use-case for this object.
    def contains(self, x):
        return True
