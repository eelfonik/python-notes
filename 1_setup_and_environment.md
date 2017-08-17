## Version management
For managing python version, I use [pyenv](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md) for now. There’s also Anancoda env dedicate to Science calculation & machine learning.

As it's always a good idea to use a isolated environment for each project, we can use [virtualenv](https://github.com/pyenv/pyenv-virtualenv) to create an env using a specific python version

## Python shell & Editors
For practice, we can simply use **IDLE** provided by python installation, or directly use Terminal app by typing `python` (use `quit()` or command + d to quit python shell and go back to terminal window).

**GoodNews**: as the IDLE is packed when you download python from website, it's not ideal for python versions installed with **pyenv**, we've got [bpython](https://docs.bpython-interpreter.org/contributing.html#getting-your-development-environment-set-up) with code highlight & hint etc right from terminals. (installed with pip)

Simply type `bpython` in terminal window to open a *colored python shell*.

IDLE knows all about Python syntax and offers *completion hints* that pop up when you use a built-in function like `print()`. Python programmers generally refer to built-in functions as **BIFs**. The `print()` BIF displays messages to standard output (usually the screen).
