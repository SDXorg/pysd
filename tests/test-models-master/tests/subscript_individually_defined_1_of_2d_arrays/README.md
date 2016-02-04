Test subscript individually defined 1 of 2d arrays
--------------------------------------------------

There are a number of different features that make up the full functionality we know as 
'subscripts'. We'll break them all into separate tests to ease the development effort.

This test exercises the syntax in which one dimension of a 2d subscript is defined using individual lines, as opposed to the `comma, semicolon;` syntax:

~~~
Rate A[One Dimensional Subscript, Column 1]=
	0.01, 0.03, 0.05 ~~|
Rate A[One Dimensional Subscript, Column 2]=
	0.02, 0.04, 0.06
	~	
	~		|

~~~

![Vensim screenshot](vensim_screenshot.png)


Contributions
-------------

| Component                         | Author          | Contact                    | Date    | Software Version        |
|:--------------------------------- |:--------------- |:-------------------------- |:------- |:----------------------- |
| test_subscript_2d_arrays.mdl      | James Houghton  | james.p.houghton@gmail.com | 11/17/15 | Vensim DSS 6.3 for Mac  |
| output.csv                        | James Houghton  | james.p.houghton@gmail.com | 11/17/15 | Vensim DSS 6.3 for Mac  |