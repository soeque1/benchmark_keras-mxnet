from setuptools import setup
from setuptools import find_packages

long_description = '''
Keras is a high-level neural networks API,
written in Python. Keras-MXNet is capable of running on top of
high performance, scalable Apache MXNet deep learning engine.

Use Keras-MXNet if you need a deep learning library that:

- Allows for easy and fast prototyping
  (through user friendliness, modularity, and extensibility).
- Supports both convolutional networks and recurrent networks,
  as well as combinations of the two.
- Runs seamlessly on CPU, one GPU and multi-GPU.

Read the Keras documentation at: https://keras.io/

Read the Keras-MXNet documentation at: https://github.com/awslabs/keras-apache-mxnet/tree/master/docs/mxnet_backend

For a detailed overview of what makes Keras special, see:
https://keras.io/why-use-keras/
Keras is compatible with Python 2.7-3.6
and is distributed under the MIT liense.
'''

setup(name='keras-mxnet',
      version='2.2.0',
      description='Deep Learning for humans. Keras with highly scalable, high performance Apache MXNet backend support.',
      long_description=long_description,
      author='Amazon Web Services',
      url='https://github.com/awslabs/keras-apache-mxnet',
      license='MIT',
      install_requires=['numpy>=1.9.1',
                        'scipy>=0.14',
                        'six>=1.9.0',
                        'h5py>=2.7.1',
                        'pyyaml',
                        'keras_applications'],
      extras_require={
          'visualize': ['pydot>=1.2.4'],
          'tests': ['pytest',
                    'pytest-pep8',
                    'pytest-xdist',
                    'pytest-cov',
                    'pandas',
                    'requests'],
      },
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
