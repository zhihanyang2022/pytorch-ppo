import gin


@gin.configurable(module=__name__)
def pick(algo=gin.REQUIRED, env=gin.REQUIRED):
    return env, algo
