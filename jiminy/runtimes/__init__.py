import os
import yaml

from jiminy.runtimes.registration import register_runtime

with open(os.path.join(os.path.dirname(__file__), '../runtimes.yml')) as f:
    spec = yaml.load(f, Loader=yaml.FullLoader)

# If you have a local repo, do something like
# export BOXWARE_DOCKER_REPO=docker.boxware.com  (this one only for boxware folks)
# docker_repo = os.environ.get('BOXWARE_DOCKER_REPO', 'quay.io/boxware')
docker_repo = os.environ.get('BOXWARE_DOCKER_REPO', 'sibeshkar')

register_runtime(
    id='gym-core',
    kind='docker',
    image=docker_repo + '/jiminy.gym-core:{}'.format(spec['gym-core']['tag']),
)

register_runtime(
    id='flashgames',
    kind='docker',
    image=docker_repo + '/jiminy.flashgames:{}'.format(spec['flashgames']['tag']),
    host_config={
        'privileged': True,
        'cap_add': ['SYS_ADMIN'],
        'ipc_mode': 'host',
    },
    default_params={'cpu': 3.9, 'livestream_url': None},
    server_registry_file=os.path.join(os.path.dirname(__file__), 'flashgames.json'),
)

register_runtime(
    id='world-of-bits',
    kind='docker',
    #image=docker_repo + '/jiminy.world-of-bits:{}'.format(spec['world-of-bits']['tag']),
    image=docker_repo + '/jiminywob:{}'.format(spec['world-of-bits']['tag']),
    host_config={
        'privileged': True,
        'cap_add': ['SYS_ADMIN'],
        'ipc_mode': 'host'
    })

register_runtime(
    id='vnc-windows',
    kind='windows',
)

del spec
