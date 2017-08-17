### string:

Strings in Python are **immutable**, which means that once a string is created, it cannot be changed.

### list:

create simple list in python:

```python
movies = ["a","b","c"]
```
=> note there’s no *declaration of type* or even things like `var`, `let`,`const`
Not so with Python: identifiers are simply names that **refer** to a data object of some type.
python的identifiers不在乎它所指的data是什么类型，它只是一个refer

=> You can use either `""` or `''` to create a string in python, up to you.

Think of Python’s list as a **high-level collection**. The type of the data items is not important to the list.

#### Two types of list in Python

- those that can change (enclosed in square brackets), like `['a','b','c']`
- immutable list (enclosed in regular brackets), like `(a,b)`, also called **tuple** => once created, the data they hold **cannot be changed** under any circumstances. Another way to think about tuples is to consider them to be a *constant list*.

#### List of list

```python
movies = [
"The Holy Grail", 1975, "Terry Jones & Terry Gilliam", 91,
	["Graham Chapman",
		["Michael Palin", "John Cleese", "Terry Gilliam", "Eric Idle", "Terry Jones"]]]
print(movies[4][1][0])

# "Michael Palin"
```

#### list slice

```python
movies = ['a','n','sds','ssd']
movies[0:3]

# ['a','n','sds']
```

#### list comprehension

Creating a new list by specifying the **transformation** that is to be applied to each of the data items within an existing list. 完全就是fp的套路

```python
movies = ['black orange', 'herzog', 'a list apart']
new_movies = [item.replace('e', '') for item in movies]
print(new_movies)

# ['black orang', 'hrzog', 'a list apart']
```

>  Much like in ES6 when we use `map`:
>  
>  ```javascript
>  const movies = ['black orange', 'herzog', 'a list apart'];
>  const new_movies = movies.map(item => item.replace('e',''));
>  console.debug(new_movies);
>  
>  // ["black orang", "hrzog", "a list apart"]
>  ```

**QUESTION**: what about `map` function in python?

It’s also possible to assign the results of the list transformation **back** onto the **original target identifier**:

```python
clean = [float(sanitize(t)) for t in ['2-22', '3:33', '4.44']]
clean

# [2.22, 3.33, 4.44]
```

**N.B.** the **key point** of list comprehension is a **transformation** function, which normally equal to a `map` in js, if the function applied to each item in the list is not transformation, but like remove some value (like `filter` in js), or create a new list from an item, list comprehesion cannot help much, we should use iteration like `for...in...` loop

### set:

Several ways to create a new set:

- create a set use factory BIF `set()` => `distance = set()`
- or directly populate a set using some data items => `distance = {10.6, 11, 8, 10.6, 'two', 7}`
- or even better, pass a list to `set()` BIF to create a set =>

	```python
	movies = ['a','b','b','c']
	movies_set = set(movies)
	movies_set
	
	# {'b', 'c', 'a'}
	```
**Important Note:** the data items in a set are **unordered** and **duplicates are not allowed**. If you try to add a data item to a set that already contains the data item, Python simply *ignores* it.



