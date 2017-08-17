- `range()` => Returns an iterator that generates numbers in a specified range on demand and as needed. 

- `isinstance(identifier, type)` =>

	BIF to check if a specific identifier holds data of a specific type 

	`isinstance(movies, list)` // true
	
	**TODO: where is an exhausted list of `type`?**

- `print(value, sep=' ', end='\n', file=sys.stdout)`
	

##### list methods:

- `len()` =>

	get length 

	`print(len(movies))` // 3
	
- `append()` => 

	add new item at the end 
	
	```python
	movies.append("d")
	print(movies)
	``` 
	// `["a", "b", "c", "d"]`
	
- `pop()` => 

	remove item at the end 

	```python
	movies.pop() // "d"
	print(movies)
	```
	// `["a", "b", "c"]`
	
- `extend()` =>

	add a whole collection to the end 

	```python
	movies.extend(["d", "e", "f"])
	print(movies)
	```
	// `["a", "b", "c", "d", "e", "f"]`
	
- `remove()` =>

	find and remove a specific item  

	```python
	movies.remove("b")
	print(movies)
	```
	// `["a", "c", "d", "e", "f"]`
	
- `insert()` =>

	insert item at a *specific slot* 

	```python
	movies.insert(1, "bla")
	print(movies)
	```
	// `["a", "bla", "c", "d", "e", "f"]`
	
- `sort()` => 'in-place' sort a list, will **replace** the original list

	
##### Operate & Iterate methods:

- `print(some_list)` 
- `sorted(some_data_object, reverse=False)` =>

	copied sort, return a new **sorted list**
	
	- default sort in ascending, pass `reverse=True` to sort descending
	- no matter what kind of data object you passed to sorted, it will alway return a **list**
		

- `for ... in ...` =>

	```python
	for movie in movies:
		print(movie)
	```
	// "a", "bla", "c", "d", "e", "f"
	
> 	*A note to myself*: in javascript there's a `for` method which I merely use, as it's not quite functional style -_- (normally use `map`, `reduce`, or `forEach`).
> 	
> 	to compare(note the `;`, `let` and different syntax):
> 	
> 	```javascript
> 	const movies = ["a", "b", "c"];
> 	for (let movie in movies) {
> 		console.log(movie);
> 	}
> 	// would normally do:
> 	movies.forEach(m => {console.log(m)});
> 	```
	
- `while` loop =>

	```python
	count = 0
	while count < len(movies):
		print(movies[count])
		count = count+1
	```
	
	this one do the same thing as the above `for` loop, but you have to worry about “state information,” which requires you to employ a counting identifier.
	
	So use `for` unless you've got a good reason to use `while`.
	


##### string BIFs

- `split(sep, maxsplit)` => return an iterator
- `find()` => return **-1** if not found, otherwise **index of** the substring
- `strip()` => remove unwanted space & tab & line-breaks from a string, see [here](https://stackoverflow.com/a/761825)


#####  factory function:
- `set()` => create a new set

