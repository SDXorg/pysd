pysd
====

System Dynamics Modeling in Python

I'd like to put together a simple library for creating and running [System Dynamics](http://en.wikipedia.org/wiki/System_dynamics)  models in python. 

There are a number of great SD programs out there ([Vensim](http://vensim.com/), [iThink](http://www.iseesystems.com/Softwares/Business/ithinkSoftware.aspx), [AnyLogic](http://www.anylogic.com/system-dynamics) ) and I should be careful not to fall into the [Not-Invented-Here](http://en.wikipedia.org/wiki/Not_invented_here) fallacy. 

What I would like to do is make a simple library that makes it easier to interface SD models with other developments in data management, data visualization, statistical machine learning, parallel computing, etc. Instead of trying to bring those features into existing SD software, I think it might make sense to bring the (relatively simple) components of SD into in a high-use programming environment where others are actively developing complementary technologies.

The best way to do this might be to take advantage of the utilities that are becoming the norm for data processing and data management: [Numpy/Scipy](http://www.numpy.org/), [Pandas](http://pandas.pydata.org/), [Matplotlib](http://matplotlib.org/) and others.

We should probably shoot for the minimum number of features:

- Stock/Flow creation
- Step function management
- Stock/Flow diagram display
- Integration

Also, we probably want to emphasize readability. 

#### Possible Feature list

- manage parameters that include units of time in a natural fashion


Similar Projects:
It would be worth seeing if any of [these](http://www.scipy.org/topical-software.html#dynamical-systems) fit the bill.
