---
title: Preparation
author-meta: LI, Qimai; Zhong, Yongfeng
date-meta: November 16, 2021, modify on September 16, 2022
lang: en
compile-command: ../panrun/panrun SetupPython.md
output:
  html:
    standalone: true
    output: SetupPython.html
    number-sections: true
    toc: true
    mathjax: true
    css: ref/github-pandoc.css
    toc-depth: 2
---

<!-- ### Watch recordings at Sharepoint
1. Speed up videos: chrome extension [Video Speed Controller](https://chrome.google.com/webstore/detail/video-speed-controller/nffaoalbilbmmfgbnbgppjihopabppdk?hl=en).
2. Subtitles: turn on subtitles at right upper corners.
<center>
<img alt="line equation" src="ref/subtitle.jpg" style="width:40%">
</center> -->

# Official Python Tutorial
This course presumes that we have learned Python before. If not, I recommend you to read the official [Python Tutorial](https://docs.python.org/3/tutorial/index.html), which is one of the best python tutorials.

Get help information in Python: we may forget the usage of a specific function sometimes. In this case, we can use a built in function `help` to get the information about a certain function. E.g., if we forget how to use `print` function, we can get help by the following command:
```python
help(print)
```
We will see the following information, showing the meaning of each argument:
```
Help on built-in function print in module builtins:

print(...)
    print(value, ..., sep=' ', end='\n', file=sys.stdout, flush=False)

    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file:  a file-like object (stream); defaults to the current sys.stdout.

    sep:   string inserted between values, default a space.
    end:   string appended after the last value, default a newline.
    flush: whether to forcibly flush the stream.
```

# Setup Python Environment
We need Python3 and several Python packages, including Numpy, Pillow, and PyTorch, to finish the programming assignments of this course. If you know how to setup and manage your Python environment, just do it in your preferred way. If not, you can follow the instructions in this section to setup Python3 development environment step by step.

## Install Anaconda
The most convenient way to setup a Python environment is to install Anaconda, a Python environment management system with lots of useful third-party packages. Anaconda already contains nearly all packages we need.

Please go to [Anaconda download page](https://www.anaconda.com/products/individual) and download a suitable version of Anaconda3 for your system. You may download the graphic installer or command line installer as you prefer. Follow the [Installation page](https://docs.anaconda.com/anaconda/install/) if needed.

<!-- ### Validate your Install
After the installation, open "Anaconda Prompt" from your Windows Start menu (or corresponding terminal in OS and Linux), and type command `conda --version` and `python --version`. If you see the following output, you have installed Conda successfully.

```bash
(base) C:\>conda --version
conda 4.8.2

(base) C:\>python --version
Python 3.7.4
``` -->

<!-- ## Install IPython
IPython is the most popular interactive shell for python. Beginners can try anything they just learn in IPython quickly. You could install ipython via following command in Anaconda Prompt.

```bash
(base) C:\>conda install ipython
...
Proceed ([y]/n)?y
...
``` -->
## Update Environment Variables

Before you run, debug your python code on a terminal, you need to run the following scripts before you using python. You can also add those scripts to `~/.bashrc` or `~/.zshrc`, just to make sure they will be run automatically every time after you start your computer.

```bash
export ANACONDA_ROOT=your_own_anaconda3_path
export PATH=$ANACONDA_ROOT/bin:$PATH
export PYTHONPATH=$ANACONDA_ROOT/lib/python3.9/site-packages:$PYTHONPATH
```
Now you can start your programming:
```bash
# run
python code/filter.py

# debug
python -m pdb code/filter.py
```

## Install PyCharm

You can also use a integrated development environment (IDE), Pycharm to run your python code. It is a very convenient tool that can help you manage, run, debug, and deploy Python programs. Go to this link [community version](https://www.jetbrains.com/pycharm/download/) to download and install it if you want.

After install PyCharm, you need to configure it to use Anaconda you installed before.

1. Open PyCharm Preference by pressing
   -PyCharm -> Preferences
2. Search "Python interpreter" in the up left corner.
3. Click "__Add interpreter__-> __Add Local interpreter__" at the right side.
4. In the new interface, choose "__Conda Environment__".
5. Select "__Interpreter__" and "__Conda executable__" as the one you installed:
    - Example on Win10:
        - Interpreter:  `C:\Users\<USERNAME>\anaconda3\python.exe`
        - Conda executable: `C:\Users\<USERNAME>\anaconda3\Scripts\conda.exe`
    - Example on macOS:
        - Interpreter: `/Users/<USERNAME>/anaconda3/bin/python`
        - Conda executable: no need to select.
6. Check "__Make available to all projects__" and click "__OK__".
7. Now come back to previous interface. Wait and then click "__OK__" again.
-->
<!--
1. Open PyCharm configuration by pressing
    - "__Ctrl__ + __Alt__ + __s__" on Windows, or
    - "__Cmd⌘__ + __,__" on macOS.
2. Search "interpreter" in left up corner.
3. Choose "⚙ -> __Add__" at right side.
4. In the new interface, choose "__Conda Environment__ -> __Existing environment__"
5. Select "__Interpreter__" and "__Conda executable__" as the one you installed:
    - Example on Win10:
        - Interpreter:  `C:\Users\<USERNAME>\anaconda3\python.exe`
        - Conda executable: `C:\Users\<USERNAME>\anaconda3\Scripts\conda.exe`
    - Example on macOS:
        - Interpreter: `/Users/<USERNAME>/anaconda3/bin/python`
        - Conda executable: no need to select.
6. Check "__Make available to all projects__" and click "__OK__".
7. Now you come back to previous interface. Wait for it finishing processing, then click "__OK__" again.
-->

After the configuation, PyCharm projects can use Anaconda as the default Python environment.

