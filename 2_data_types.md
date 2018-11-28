## Data types

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

# access list data using index
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

Creating a **new list** by specifying the **transformation** that is to be applied to **each of** the data items within an **existing list**. 完全就是fp的套路

In the codes below: 

- *new list* is `new_movies`
- *tranformation function* is `item.replace('e', '')`
- *each items* is named `item`
- *existing list* is `movies`

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

_Several ways to create a new set_:

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


### dictionary:

_Several ways to create a new dict_:

- factory BIF `dict()` => `clees = dict()`
- using curly braces => `clees = {'Name': 'kino', 'Age': 'unknown', 'Life': ['outer space', 'coding', 'sleep']}`

	**N.B.** the tricky part is, when using curly braces to create a variable, if the data is *empty* or *structured using `key: value`* (see above) , the type is `dict`, and when the data is just *a list of items*, it becomes `set`.
	
_Accessing data inside dict:_

```python
clees['Name']
# 'kino'

clees['Life'][-1]
# 'sleep'
```

_Add new items to a dict:_

```python
clees['Stop'] = "why, so, hard"
clees
# {'Name': 'kino', 'Age': 'unknown', 'Stop': 'why, so, hard', 'Life': ['outer space', 'coding', 'sleep']}
```
**N.B.** the added item **does not maintain insertion order**, the dictionary maintains the *associations*, not the *ordering*.

#### dict comprehension (of course)

It takes the form `{key: value for value in iterable}`

```python
movies = ['black orange', 'herzog', 'a list apart']
new_movies_dict = { movies.index(item): item for item in movies }
print(new_movies_dict)

# {0: 'black orange', 1: 'herzog', 2: 'a list apart'}
```

Or a more general pattern use `{key:value for (key,value) in dictonary.items()}`
```python
dict1 = {'a': 3, 'b': 5, 'c': 6}
new_dict_1 = {k*2:v for (k,v) in dict1.items()}
#{'aa': 3, 'bb': 5, 'cc': 6}
new_dict_2 = {k: v*2 for (k,v) in dict1.items()}
#{'a': 6, 'b': 10, 'c': 12}

for (k, v) in dict1.items():
  print(k,v)
# a 3
# b 5
# c 6
```
