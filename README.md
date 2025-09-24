# Keyboard Switcher

Automatically detect and switch the language (_En<->He<->Ru_) for **Windows**.

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The main idea was to resolve the annoying situation, when you are working with different
languages (in our case it is russian, hebrew and english) and all the time have to switch between
them. Like for example when you are switching between the browser (english layout for the url) 
and text chat (hebrew or russian layout)<br>

To get the result we have collected a large dataset, based on english, hebrew and russian dictionaries.<br>
We have trained a DL model with a relatively simple LSTM architecture to detect the 
language properly.<br>

Pytorch model trained on GPU but can be used on CPU without any problem<br>
The accuracy is above 99% on test set.

The program is operated as a background task, that is initiated on mouse click.
It detects the input language (most probable) and corrects the input, getting into
the sleep mode until the next mouse click.

## Installation

The package can be running with python as follows:
```
# installing and running with python:
git clone https://github.com/kertser/KeyboardSwitcher.git
cd KeyboardSwitcher
# pay attention to the unnecessary dependencies in requirements, 
# needed for training the model only
pip install -r requrements.txt
python3 main.py
```
Alternatively, the software can be operated as Windows Program
(shall be downloaded and saved into some directory)

## Usage
Just run the program, make a mouse click, type and enjoy.<br>
**If there are some bugs, let me know and I will fix'em...**

## Contributing

Me, myself and I. :) <br>
Special thanks to the open language vocabularies, used as a dataset for training

## License
Free to use, change and add at your will.<br>
Dataset and jupyter notebook for model training/evaluation is added as well
