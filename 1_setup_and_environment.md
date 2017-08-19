## Version management
- [pyenv](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md) for managing python version. There’s also Anancoda env dedicate to Science calculation & machine learning.

- [virtualenv](https://github.com/pyenv/pyenv-virtualenv) As it's always a good idea to use a isolated environment for each project, we can use *pyenv-virtualenv* to create an env using a specific python version

- [docker](https://docs.docker.com/docker-for-mac/#explore-the-application-and-run-examples) Should be considered using with vitualenv?

	```bash
	docker run -d -p 4000:80 --name webserver nginx
	# start a new docker container named 'webserver' from ngnix image
	# port remapping of 4000:80, 4000 means the port used when publishing to host OS, and 80 is what the container EXPOSE within the Dockerfile
	# -p for publish, -d for detached mode (running in background)
	# the name is important as you can refer to the container later
	
	docker stop/start webserver
	# will stop/start the container
	```
	
	- use `docker ps` to list all active containers, and `docker ps -a` to list all containers
	- `docker rm -f webserver` will remove the 'webserver' container, but not the `ngnix` image that it created from.
	- use `docker images` to list all images available, and `docker rmi imageName|imageID` to delete an image 

## Python shell & Editors
For practice, we can simply use **IDLE** provided by python installation, or directly use Terminal app by typing `python` (use `quit()` or command + d to quit python shell and go back to terminal window).

**GoodNews**: as the IDLE is packed when you download python from website, it's not ideal for python versions installed with **pyenv**, we've got [bpython](https://docs.bpython-interpreter.org/contributing.html#getting-your-development-environment-set-up) with code highlight & hint etc right from terminals. (installed with pip)

Simply type `bpython` in terminal window to open a *colored python shell*.

IDLE knows all about Python syntax and offers *completion hints* that pop up when you use a built-in function like `print()`. Python programmers generally refer to built-in functions as **BIFs**. The `print()` BIF displays messages to standard output (usually the screen).
