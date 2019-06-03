from setuptools import setup, find_packages

setup(name='jiminy',
      version='0.21.5',
      description="Jiminy is an imitation learning library.",
      url='https://github.com/sibeshkar/jiminy',
      author='Boxware',
      author_email='sibesh96@gmail.com',
      packages=[package for package in find_packages()
                if package.startswith('jiminy')],
      install_requires=[
          'numpy>=1.16.2',
          'autobahn>=0.16.0',
          'docker-py==1.10.3',
          'docker-pycreds==0.2.1',
          'fastzbarlight>=0.0.13',
          'go-vncdriver>=0.4.8',
          'Pillow>=3.3.0',
          'PyYAML>=5.1',
          'six>=1.10.0',
          'twisted>=16.5.0',
          'ujson>=1.35',
          'opencv-python>=4.0'
      ],
      package_data={'jiminy': ['runtimes.yml', 'runtimes/flashgames.json']},
      tests_require=['pytest']
      )
