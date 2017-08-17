### read data from files
- 3 BIFs to **open, read & close** file

	```python
	# open some txt file
	data = open('path/to/somefile.txt')
	# print out line by line on using *readline()* (this time the first line)
	print(data.readline(), end='')
	# go back to first line
	data.seek(0)
	
	# print all lines in this file
	for each_line in data:
		print(each_line, end='')
	# basically it means the *data* variable created by opening a file
	# is an iterable object (an iterator), composed by each line of that file
	
	
	# after all done, close file	
	data.close()
	
	```


### handle exceptions

Python interpreter displays a **traceback** followed by an *error message* when something went wrong. Runtime errors are called **exceptions**

**Try first, then recover** => Rather than adding **extra code and logic to guard against** bad things happening, Python’s **exception handling** mechanism lets the error occur, spots that it has happened, and then gives you an opportunity to recover.

![image](https://www.dropbox.com/s/9jtaopefd1owyl9/Screen%20Shot%202017-08-16%20at%207.28.49%20PM.png?raw=1)

**TODO: shoule I also use this pattern in javascript? (basically use `try... except...`( `try... catch...` in js) instead of `if... else...`**

- `try...except...` statement with `pass` statement (just like js `return null`)

   ```python
	try:
		# some code here
	except ValueError:
		# some error handling code here or
		pass
	finally:
		#some final code that no matter what you want to execute
	```
	
	- Note it's considered as an **anti-pattern** to handle exception without types, as it might **silently ignore** runtime errors.
	We should also use `finally` statement to clean up the `try... except...` block, see [here](http://docs.quantifiedcode.com/python-anti-patterns/correctness/no_exception_type_specified.html)
	- Actually as we have to be specific about the error types, the only advantage of using `try... catch...` instead of `if... else...` is that: we don't have to write the `if` condition. Just the same reason as we prefer `for...in...` over `while`.

- `with` statement is specially useful to file operations with exceptions:

	```python
	# before 
	try:
		data = open('some.txt')
		for each_line in data:
			print(each_line, end='')
	except FileNotFoundError as err:
		print('File error: ' + str(err))
	finally:
		if data in locals():
			data.close()
			
	# after
	try:
		with open('some.txt') as data:
			for each_line in data:
				print(each_line, end='')
	except FileNotFoundError as err:
		print('File error: ' + str(err))
		
	# note we don't have to add a `finally` statement just to check & close file
	```
	
	The with statement takes advantage of a Python technology called the **context management protocol**.


### save data to files

- the BIF `open()` has actually a 2nd argument to specify the mode to use (much like vim editor)
	- `open('some.txt', 'r')` is for **read** only, which is default, raise a `FileNotFoundError` if file not exits
	- `open('some.txt', 'w')` is for **write** a file, note if you open a file with existing content, this mode will **clear! all content**, and if the file does not exist already, it will create one
	- `open('some.txt', 'a')` is for **append** content to a file, if the file does not exist already, it will create one
	- `open('some.txt', 'r+')` is for read & write, raise a `FileNotFoundError` if file not exits
	- `open('some.txt', 'w+')` is for **read & write** a file, note if you open a file with existing content, this mode will **clear! all content**, and if the file does not exist already, it will create one

- then the BIF `print()` can take another arguement `file` which specify the output space (by default it's using `sys.stdout`, defined in standard library `sys`, and the standard output is always the screen) => `print(data, file=output_object)`

	Combine thoes two & the technique of `with` statement, we can write like:
	
	```python
	movies = ['a','b','c']
	try:
		with open('new/text.txt', 'w') as out:
			print(movies, file=out)
	except OSError as err:
		print('File error: ' + str(err))
	```
