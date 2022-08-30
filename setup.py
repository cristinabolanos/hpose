from setuptools import setup


def requirements() -> list:
    retval = []
    with open('requirements.txt') as fr:
        for line in fr:
            if line.isspace() or line.startswith('#'):
                continue
            retval.append(line.strip())
    return retval


setup(name='hpose',
      version='0.0.0',
      author='Cristina Bolaños Peño',
      author_email='cristinabope@gmail.com',
      license='GNU General Public License v3',
      description='Human pose detection library (Python3)',
      long_description='This is a Python3 library ' +
      'which allows one to extract human joints ' +
      'position from a media file (or camera ' +
      'device stream).',
      py_modules=['hpose'],
      package_data={'hpose': ['README.md', 'LICENSE']},
      install_requires=requirements(),
      classifiers=['Development Status :: 3 - Alpha',
                   'License :: OSI Approved :: ' +
                   'GNU General Public License v3 or later (GPLv3+)',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3 :: Only',
                   'Programming Language :: Python :: 3.10',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.7',
                   'Topic :: Scientific/Engineering',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence',
                   'Topic :: Software Development',
                   'Topic :: Software Development :: Libraries',
                   'Topic :: Software Development :: Libraries' +
                   ' :: Python Modules'],
      keywords=['machine-learning', 'tensorflow', 'tflite',
                'human-pose-detection', 'pose-detection',
                'embedded-systems', 'rpi4', 'coral-dev-board'])
