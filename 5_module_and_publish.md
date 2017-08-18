
### module and import

1. create a file called `something.py`, where has `def some_func`
2. then in any other py file in the **same directory**, or python shell with **cwd** as the **root directory** of the file to import, you can use `from something import some_func`.

**TODO: Should find more about how import works**


### Prepare a module to upload & publish
1. create a file called `nester.py` inside a folder named 'python_practice', inside that file, put the `print_items` function that we write with some comments
2. in the **same folder**, create a `run.py` or whatever name you want, which import setup from `distutils.core` utility, and add metadata to your module:

```python
# run.py
from distutils.core import setup
setup(
		name ='nester',
		version = '0.0.1',
		py_modules = ['nester'],
		author = 'kino',
		author_email = 'kinoflee@gmail.com',
		url = 'http://k.42web.co',
		description = 'A simple printer of nested lists',
	 )
```

3. Open a terminal window, cd to the folder contains those 2 files ( 'python_practice' in our case ), then first run `python3 run.py sdist`,
then `sudo python3 run.py install`.

4. this will create `manifest` & dist & build folder inside the target folder
5. then from the *python shell*, you can use:
	- `import nester`, then `nester.print_items(movies)` (*note* here as we calling the function from python shell, code in your main Python program (and within IDLE’s shell) is associated with a **namespace** called `__main__`. So we have to specify the **namespace** `nester` given by the *setup -> name* to find the function. 
	- `from nester import print_items`, then use directly `print_items(movies)` （*note* But you need to be careful. If you already have a function called `print_items` defined in your **current namespace**, the specific import statement **overwrites** your function with the imported one, which might not be the behavior you want.)

