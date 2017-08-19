## MVC on backend

- Model

Data storage & processing code

- View

format & display user interface => which would interestingly become the front-end's **Model** layer :)

- Controller

glue your webapp together and provide its business logic (can be really fat)



### web development
Instead of generating html template, we could prepare some Restful APIs in the **view** layer of the backend.

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
avoid race condition

python3 is packed with a SQLite...
