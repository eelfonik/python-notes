## The OO world

### Class
Class is to enable you to bind the data and it's function together.
Once the definition of a **class** is done, it has several notions:

- **instances** : create (or instantiate) *data objects*, which inherit their characteristics from the class, is referred to as the class's *instances*.
- **methods** : *functions* is often referred to as the class’s *methods*
- **attributes** : *data* is often referred to as the class's *attributes*

class is a *custom factory function* allows user to create *instances*. The *methods* are shared, the *attributes* are not.


#### define a class using `class` keyword
define a class with a special method `__init__()` to control how objects are initialized with given data (or aks attributes)

```python
class Athlete:
	def __init__(self, value=0):
		self.thing = value
	def get_value(self):
		return len(self.thing)
		
# ...some codes after

a = Athlete(4)
a.get_value()
# 1
	
```

In fact, when use `a = Athlete(some_data_object)` to create a new instance, it means `Athlete().__init__(a, some_data_object)`, so the `self` is important as it then refers to the *instance created*, not the *class*. The target identifer is assigned to the self argument.

As a result, self needs to be the *first argument* to **every object method**

_Some thoughts:_

- Normally we should provide a class with *well deconstructed data* but **not** *computed data*, instead of directly put a file name to let it handle the reading process.
- The whole point of using class is to **hide implementation details** from user, so  try to add methods inside class, not expose the assumption of what kind of data you want.


#### Inheritance

A much useful way is to inherit from built-in data structure classes like `list`, `set` or `dict`, then you can get all the existing methods in these classes with your custom ones. It's called **subclass**

```python
class AthleteList(list):
	def __init__(self, name, values = []):
		# note this line init list with an empty []
		# ???why?
		list.__init__([])
		self.name = name
		# don't use this 
		# self.values = values
		self.extend(values)
	def get_values_len(self):
		# note here self has became a data structure
		return len(self)
		
# Then of course we can use a.get_values_len method with our data
# but also we have all the built-in methods of list like append

a = AthleteList('Jonny', ['a','b','c'])
a.append('some text')

for attr in a:
	print(a.name + ' has ' + attr)
# "jonny has some text"
# "jonny has a"
# "jonny has b"
# "jonny has c"

# we can access the attr appended directly using list.append() method here
# together with the extend values list we passed when instantiate the 'a' instance

a.get_values_len()
# 4
```

**TODO: we can also inherit from multiple built-in data structures.**

**Note:** favour *composition* over *inheritance*


### Decorator

- `@staticmethod`
- `@property` => @property decorator allows *a class method* to appear like an *attribute* to users of the class. So instead of calling `.somemethod()`, you should just use `.somemethod`, this is useful when the method is not changing the object's data but just *convert the attributes*.