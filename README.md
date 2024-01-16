# Contact-IL -- Top Level Repository

The following is a meta-repository containing all of the packages required to run code for the Contact/STS Imitation Learning project.

# Internally Developed Libraries in this Package
- `contact-il`: The contact_il package that runs most of the imitation learning code, including demonstration collection and testing, as well as config parameters to be used with learning code from `place_from_pick_learning`.
- `panda-polymetis`: An interface between Meta's [polymetis](https://facebookresearch.github.io/fairo/polymetis/index.html) and the environments used in this package.
- `contact-panda-envs`: Gym environments for contact/tactile tasks to be used with a simulated or real Franka Panda.
- `sts`: A library for processing sts data.
- `pysts`: Sensor interfacing code that can be used to acquire STS data.
- `transform_utils`: A set of generic utilities for working with SO3 and SE3 transformations.
- `place_from_pick_learning`: Code based on another project used for training policies.

# Third Party Dependencies
Installation instructions for these packages are all found in the `contact-il` sub-package.

- [polymetis](https://github.com/facebookresearch/fairo/tree/main/polymetis)
- [torchvision](https://github.com/pytorch/vision)
- [inputs](https://github.com/trevorablett/inputs)

# Data
The training and testing datasets corresponding to this project are currently closed-source and owned by Samsung.