# QuickSeg

Small UI application to remove shifts from video microscopy

## Installation
### Prerequisites
First, make sure to have python3 installed (for windows especially, on Linux and MacOS it is normally already installed).

To test if you have python3 on windows :

Press "Super + R" (Super is the "windows" key between Ctrl and Alt on the bottom left of the keyboard)
Type cmd
Press Enter
Type python
You should see something like :

```
Python 3.8.5 (default, Jul 28 2020, 12:59:40) 
[GCC 9.3.0] on windows
Type "help", "copyright", "credits" or "license" for more information.
>>> 
To leave, type exit() then exit
```
If python3 is not installed on windows, go on the Microsoft Store and install python3.9

### For windows
Make sure you have all the prerequisite
Download the zip file (green button "code")
Unzip the file

Then to run, double click on run.bat

If you want a Desktop shortcut, right click on run.bat, click on Send To (Envoyer vers) and Desktop shortcut

### For MacOS / Linux
Run in a shell

To install

```
# For MacOS only
xcode-select --install

git clone https://github.com/Ambistic/QuickSeg.git
```

To run

```
# be sure to be in the QuickSeg folder (use `pwd`)
source tifenv/bin/activate
python3 main.py
```

To update

```
# be sure to be in the QuickSeg folder (use `pwd`)
git pull origin main
```
