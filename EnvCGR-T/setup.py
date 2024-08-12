from setuptools import setup

    # This environment requires the CoppeliaSim program
    # to be installed with the graphical model of the
    # Khepera IV robot and the scene of position control    


setup(
    name="EnvCGR_T",
    version="0.0.1",
    install_requires=["gymnasium"],
)