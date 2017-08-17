## Function
A function in Python is a **named suite of code**, which can also take an optional list of arguments if required.

use keyword `def` to define a function

```python
def function_name(list_of_arguments):
	function code suite
```

EX: a *recursive* function to print out all the items in `movies` list

```python
def print_items(item_list):
	for item in item_list:
		if isinstance(item, list):
			print_items(item)
		else:
			print(item)

print_items(movies)
```

*N.B.* Use `"""bla bla bla"""` or `'''bla bla bla'''` or `# bla bla bal` to add comment to py code

#### optional arguments

Just like in ES6 (A little bit different)!!!

- If an argument has no default value, it's **required** (this is different from ES6, as argument without default value can also be optional, will just throw an error if you try to use an undefined argument)
- give the argument a default value will make it **optional**.

EX:

```python
def print_items(item_list=['a','b','c']):
	for item in item_list:
		if isinstance(item, list):
			print_items(item)
		else:
			print(item)
			
# call with a given list
print_items(movies)
# a list of items inside `movies` list

# call without a list will output the default
print_items()
# 'a'
# 'b'
# 'c'
```

**QUESTION: How to deal with multiple optional arguments when call a function, when you don't want to specify certain optional arguments?**
Aka like in ES6, instead of passing `someFunc(arg1, arg2=0, arg3)`, passing an object as named arguments `someFunc({arg1, arg2: 0, arg3})`?

**ANSWER:** if a python function has optional variables with default value, you don't have to worry about the *position of argument* when calling it, for example:

```python
def some_func(arg1, arg2=False, arg3=3):
	# some function code
	
some_func('blabla', arg3=1)

# as ALL optional arguments are NAMED, you can use the name to pass args when calling it.
```

## Variables

Python has very powerful & convenient way to assign values to variables:

- no assignement keyword => `movies=['a','n','s']`
- direct deconstruction of variable names =>
	
	```python
	(a,b)=['sds','sdas']
	# Note here `(a,b)` is an immutable list!!!
	# AKA tuple
	
	print(a) # 'sds'
	print(b) # 'sdas'
	```
	this means the values can be any iterator object (a list, **an iterator returned** by a BIF like `str.split(';')`
	

