def shapeMatcher(shape1, shape2):
    shape1 = list(shape1)
    shape2 = list(shape2)
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] is None:
            continue
        if shape1[i] != shape2[i]:
            return False

    return True
