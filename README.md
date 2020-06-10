# Bumblebee
This repo is for latest development of sketch recognition/SBIR with transformer.
## Dependencies
Look at dependencies for list of python requirements, specifically the two files [Dockerfile](dependencies/Dockerfile) and [requirements.txt](dependencies/requirements.txt). You don't have to use Docker though.

## Read/write/convert sketch formats
Look at [skt_tools.py](utils/skt_tools.py) for list of methods reading/writing/visualising svg/stroke-3 sketches.

Also look at prep_data/sketchy for step-to-step pre-processing svg files in sketchy.
