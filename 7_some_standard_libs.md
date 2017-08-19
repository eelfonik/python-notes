- `os`:
	- `os.getcwd()` => get current working directory
	- `os.chdir('../some/path')` => change directory
	- `os.path.exists('sketch.txt')` => check if a file exits
- `distutils.core` => `distutils.core.setup` can be used to set python module metadata in a `setup.py` file, then run that file from terminal like `python3 setup.py sdist` & `sudo python3 setup.py install`
- `sys` => `sys.stdout` specify the default standard output space (screen)
- `pickle`: a library used to handle file reading(`load()`) & writing(`dump()`). The only requiremnt is all files should be in **binary access mode**

	```python
	import pickle
	
	# note the `b` after `w` in the 2nd argument, is tells using binary mode
	with open('mydata.xxx', 'wb') as some_out_object:
		pickle.dump(['a','list','hooray'], some_out_object)
		
	# some code....
	
	# later
	
	with open('mydata.xxx','rb') as input_data:
		a_list = pickle.load(input_data)
	
	print(a_list)
	# ['a','list','hooray']
	```
	
	**N.B.** as `pickle` is using a python specific protocol to process files, it may make the output files (especially the text files) looks weird on other program, it really shines when you load some previously pickled data into another program.
	
- `string` => has sth like `string.Template`
- `http.server` used to spin up a simple web server =>

	```python
	from http.server import HTTPServer, CGIHTTPRequestHandler
	port = 8080
	httpd = HTTPServer(('', port), CGIHTTPRequestHandler)
	print("Starting simple_httpd on port: " + str(httpd.server_port))
	httpd.serve_forever()
	```
- `glob` => `glob.glob('data/*.txt')` create a list of filenames inside a directory
	
- `cgi` => read data from web server `cgi.FieldStorage()`
- `cgitb` => send the *standard errors* to browser `cgitb.enable()`
- `json` => to handle json format

	- `json.dumps()` => create a stringed version of a Python type
	- `json.loads()` => create a Python type from a JSON string

