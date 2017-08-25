## MVC on both ends

**M**odels **V**iew and **C**ontroller

-
**Models** is basically some files define the data structures??? => yes, Data storage & processing code. 

on _backend_ it's define the **schema & structure** inside the database

on _Frontend_, it's define the **shape of state in store** actually.

##### Redux on server to better get data from server to fill initialState on client side
See [this video tutorial](https://egghead.io/lessons/javascript-redux-normalizing-the-state-shape)

-

**View** used to format & display user interface.

_As the MVC pattern can be applied **both** on client side or server side._

The *View layer of server side* can actually be the *Model layer of client side* :)

-

**Controller** glue your webapp together and provide its business logic (can be really fat). Normally it's how we manipulate the **data**.

On _backend_, it can interact with database, as do **query(find)** from database or **save** to database or **delete(drop)** from database, and return some callback functions accroding to the results it gets, so it's basically defined the **functions** to manipulate the database.

On _Frontend_, We can call them **actions** & **reducers**.

*Question*: _is controller where we designed our RESTful API?_

-

**Some side note on server router**: this is where the API endpoints are defined (like '/signin'), and framework like **Express** provides the common methods like *post*, *get*, *delete*, etc, to accept client requests, then send response using the **functions** defined by **controller**.

*Question*: router should be something to navigate from one point to another inside apps no? Why it feels like API endpoint?????


-
### web development
Instead of generating html template, we could prepare some Restful APIs in the **view** layer of the backend... but let's see how web used to work :)

This dynamic content generation process has been standardized since the early days of the Web and is known as the **Common Gateway Interface (CGI)**. Programs that conform to the standard are often referred to as *CGI scripts*.

the things going on inside `cgi-bin` directory ?

For CGI script to be excuted in _*NIX system_ (most of the servers are):

1. Set the executable bit for your CGI using the chmod +x command.

	`chmod +x generate_list. py`

2. add `#! /usr/local/bin/python3` at the very top of every python CGI script. 

**N.B.** you only have to do this to the files that actually need to generate and return something when excuted by **external actions** like *url clicked*, that's why we need to add that very first line: to tell the bash to use *which command* to run the script. All other `.py` files as **program** are good as they are.

That means... you are *not limited* to use python to write CGI scripts even you're using python as the backend language. (But that'll be much easier since you already use python :)) 
 
> 对于一个生成static html的后台而言，html file里需要请求后台生成内容的操作（例如一个`<a>`或者一个form submit），可以直接把对应的CGI script的url放在`href`或者`form action`里。
> 
> 如果是一个form action, python有自带的库`cgi`, with functions like `cgi.FieldStorage()`, that can be accessed pretty much like a dict that stored `k:v` where `v` is list alike: `cgi.FieldStorage()['name'].value`.
> 
> 直接生成完整的html页面的后台多半会存储一些类似`header.html`, `footer.html`之类的不完整的template,然后CGI script的操作一般是读取这些文件，动态填充内容，用`print()`拼到一起，再输出成一个完整的html字符串-_-

_CGI tracking_

The web server will capture the `stdout` of the CGI script, and ignore the `stderr`. But sometimes it's useful to also capture errors when the script is errored during *dev phase*. Python has a standard library called `cgitb` to handle this. Once enabled, it will output useful details about errors in browser.


### database
Avoid race condition

python3 is packed with a SQLite... & a `sqlite3` standard lib

#### Python Database API

![image](https://www.dropbox.com/s/r3msahv780gklvf/Screen%20Shot%202017-08-19%20at%202.27.41%20AM.png?raw=1)

```python
import sqlite3connection = sqlite3.connect('test.sqlite')
cursor = connection.cursor()
cursor.execute("""SELECT DATE('NOW')""")
connection.commit()
# or connection.rollback()connection.close()
```

##### _database design_

Define your schema and create some tables (sql) 

```sql
CREATE TABLE athletes (	id INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE NOT NULL,
	name TEXT NOT NULL,	dob DATE NOT NULL )
	
CREATE TABLE timing_data (	athlete_id INTEGER NOT NULL,	value TEXT NOT NULL,	FOREIGN KEY (athlete_id) REFERENCES athletes)
	
# the FOREIGN KEY here has a reference to `athletes` table
```

_for nosql database like `mongodb`, see the section below_.


-

### Back to the free-tracker app I build
##### Question about session cookie and localStorage:
What's the relationship with `connect sid` inside cookie (set by express session) & the state inside `localStorage` (set by redux)? I tried to keep them on sync in code, but since you can modify `localStorage` directly from dev tool, it should not be trusted hence should not be used to decide which page to show?
##### Mongodb data modeling
[officiel doc](https://docs.mongodb.com/manual/core/data-modeling-introduction/)

[Intro to 1-N data design](http://blog.mongodb.org/post/87200945828/6-rules-of-thumb-for-mongodb-schema-design-part-1)

So for the `userInfo` that will needed for every invoice, I embeded it directly inside `user`, since it's a 1-1 relationship (But as I need to get that info for every invoice, would it be slower to query?) 


