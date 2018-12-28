## Version management
- [venv](https://docs.python.org/3/library/venv.html)
  1. first `brew install python` to install latest python3 version.

  2. Then run `python3 -m venv ~/path/to/env` to create directories inside the directory, containing a copy of the Python interpreter, the standard library, and various supporting files.

  3. source to active the env `source ~/path/to/env/bin/activate`
  
  4. for VS code intergation, we need to put the venv directory as a sub-directory under project folder, then specify the python runtime by `command + shift + p` => `Select Interpreter`, choose from the dropdown the one with `venv` specified. ![alt text](./assets/python-interpreter-vscode.png)

Detailed doc [here](https://docs.python.org/3/tutorial/venv.html)

- [pyenv](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md) for managing python version. There’s also Anancoda env dedicate to Science calculation & machine learning. 

- <del>[virtualenv](https://github.com/pyenv/pyenv-virtualenv) As it's always a good idea to use a isolated environment for each project, we can use *pyenv-virtualenv* to create an env using a specific python version. </del>

- [docker](https://docs.docker.com/docker-for-mac/#explore-the-application-and-run-examples) Should be considered using with vitualenv?

## Python shell & Editors
For practice, we can simply use **IDLE** provided by python installation, or directly use Terminal app by typing `python` (use `quit()` or command + d to quit python shell and go back to terminal window).

**GoodNews**: as the IDLE is packed when you download python from website, it's not ideal for python versions installed with **pyenv**, we've got [bpython](https://docs.bpython-interpreter.org/contributing.html#getting-your-development-environment-set-up) with code highlight & hint etc right from terminals. (installed with pip)

Simply type `bpython` in terminal window to open a *colored python shell*.

IDLE knows all about Python syntax and offers *completion hints* that pop up when you use a built-in function like `print()`. Python programmers generally refer to built-in functions as **BIFs**. The `print()` BIF displays messages to standard output (usually the screen).
