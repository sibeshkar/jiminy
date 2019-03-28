from setuptools import setup, find_packages

setup(name='jiminy',
      version='0.21.5',
      description="Jiminy: a software platform for measuring and training an AI's general intelligence across the world's supply of games, websites and other applications.",
      url='https://github.com/sibeshkar/jiminy',
      author='Boxware',
      author_email='jiminy@boxware.com',
      packages=[package for package in find_packages()
                if package.startswith('jiminy')],
      install_requires=[
          'autobahn>=0.16.0',
          'docker-py==1.10.3',
          'docker-pycreds==0.2.1',
          'fastzbarlight>=0.0.13',
          'go-vncdriver>=0.4.8',
          'gym>=0.8.1',
          'Pillow>=3.3.0',
          'PyYAML>=5.1',
          'six>=1.10.0',
          'twisted>=16.5.0',
          'ujson>=1.35',
      ],
      package_data={'jiminy': ['runtimes.yml', 'runtimes/flashgames.json']},
      tests_require=['pytest'],
      extras_require={
          'atari': 'gym[atari]',
      }
      )
