Test Subscript 3D Arrays
========================

There are a number of different features that make up the full functionality we know as 'subscripts'. We'll break them all into separate tests to ease the development effort.

3d subscripts are challenging in vensim because we can no longer use the `comma, semicolon; comma, semicolon;` syntax to define the elements of subscripted data.
 
From the [vensim helpfiles](http://www.vensim.com/documentation/22070.htm): 
>If you have a constant with more than 2 subscripts it will be necessary to write multiple equations for it with each equation having no more than 2 subscript ranges.  For example:
>
~~~
sex : female,male ~~|
ini population[country,blood type,female] = 1,2,3,4;5,6,7,8;
         9,10,11,12; ~~|
ini population[country,blood type,male] = 1,2,3,4;5,6,7,8;
         9,10,11,12; ~ Person ~|
~~~

![Vensim screenshot](vensim_screenshot.png)


Contributions
-------------

| Component                         | Author          | Contact                    | Date    | Software Version        |
|:--------------------------------- |:--------------- |:-------------------------- |:------- |:----------------------- |

