# Ossify

[![PyPI version](https://img.shields.io/pypi/v/ossify)](https://pypi.org/project/ossify/)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://www.csdashm.com/ossify/)

Ossify is a library to work with neuronal morphology, with a focus high resolution synaptic level reconstructions. Importantly, it aims to solve the challenge of working with objects that have multiple representations, such as meshes and skeletons, decorated with annotations like synapses and allows users to easily convert features and metadata between them. In addition, it provides tools to load, manipulate, visualize, and analyze neuron morphologies.

Ossify is built around the concept of a Cell, which is a container for multiple Layer objects that represent different aspects of the morphology. Surface meshes capture detailed geometry, skeletons provide a rooted tree structure, and point cloud annotations like synapses can decorate different parts of the morphology. It's aim is to be flexible, but with a focus on datasets like MICrONS or FlyWire that are hosted in CAVE.

These are very early days, but the code is already very usable and roughly feature-complete with previous tools like MeshParty. The main thing missing right now is feature generation. Documentation is largely produced by Claude Code, so your mileage may vary, and I'm slowly going through it.

