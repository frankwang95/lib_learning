from setuptools import setup, find_packages


setup(
	name='lib_learning',
	version='0.0.0',
	author='Frank Wang',
	author_email='fkwang@uchicago.edu',
	packages=find_packages(),
	include_package_data=True,
	install_requires=[
		'numpy',
		'tensorflow==1.6.0',
		'plotly'
	]
)
