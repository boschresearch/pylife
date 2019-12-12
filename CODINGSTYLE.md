# pyLife coding style guidelines

## Introduction

One crucial quality criteria of program code is maintainability. In order to
maintain code, the code has to be written clearly so that it is easily readable
to someone who has not written it. Therefore it is helpful to have a consistant
coding style with consistant naming conventions. However the codings style
rules are not supposed to be strict rules. They can be ignored if there are
good reasons to ignore them.

As we are programming in Python we vastly stick to the [PEP8 coding style
guide][1]. That document is generally recommendable to python python
programmers. This document therefore covers only things that go beyond the
PEP8.


## Naming conventions

By naming conventions the programmer can give some indications to the reader of
the program, what an identifier is supposed to be or what it is refering
to. Therfore some consistency guidelines.


### Descriptive naming styles

Class names are usually short and a single or compund noun. For these short
names we use the so called `CamelCase` style:
```python
class DataObjectReader:
...
```

Function and variable names can be longer than class names. Especially function
names tend to be actual sentences like:
```python
def calc_all_data_from_scratch()
```
These are way more readable in the so called `lowercase_with_underscores`
style.


### Variable names

Variable names can be shorter as long as they are local. For example when you
store the result of a function in a variable that the function is finally to
return, don't call it `result_to_be_returned` but only `res`. A rule of thumb
is that the name of a variable needs to be descriptive, if the code part in
which the variable is used, exceeds the area that you can capture with one eye
glympse.


### Class method names

There are a couple of conventions that make it easier to understand an API of a
class.

To acccess the data items of a class we usually use getter and setter
functions. The setter functions are supposed to do sanity checks to ensure that
the internal data structure of the class remains consistant. The following
example illustrates the naming conventions:
```python
class ExampleClass:
	def __init__(self):
		self._foo = 23
		self._bar = 42
		self._sum = None

	def foo(self):
		''' getter functions have the name of the accessed data item
		'''
		return self._foo

	def set_foo(self, v):
		''' setter functions have the name of the accessed data item prefixed
			with `set_`
		'''
		if v < 0: # sanity check
			raise Exception("Value for foo must be >= 0")
		self._foo = v

	def calc_sum_of_foo_and_bar(self):
		'''	class methods whose name does not imply that they return data
			should not return anything.
		'''
		self.sum = self.foo + self.bar
```

### Data encapsulation

One big advantage for object oriented programming is the so called data
encapsulation. That means that items of a class that is intended only for
internal use can be made unaccessible from outside of the class. Python does
not strictly enforce that concept, but in order to make it clear to the reader
of the code, we mark every class method and every class member variable that is
not meant to be accessed from outsid the class with a leading underscore `_`
like:
```python
class Foo:

	def __init__(self):
		self.public_variable = 'bar'
		self._private_variable = 'baz'

	def public_method(self):
	...

	def _private_method(self):
```


## Structuring of the code

### Object orientation

Usually it makes sense to compund data structures and the functions into
classes. The data structures then become class members and the functions become
class methods. This object oriented way of doing things is recommendable but
not alsways necessary. Sets of simple utility routines can also be autonoumous
functions.

As a rule of thumb: If the user of some functionality needs to keep around a
data structure for a longer time and make several different function calls that
deal with the same data structure, it is probably a good idea to put everything
into a class.

### Functions and methods

Functions are not only there for sharing code but also to devide code into
easily overseeable pieces. Thefore functions should be short and sweet and do
just one thing. If a function does not fit into your editor window, you should
consider to split it into smaller pieces. Even more so, if you need to scroll
in order to find out, where a loop or an if statement begins and ends.

Rule of thumb: The best functions fit on your screen together with their
docstring, so that you can see what they are doing and how they are doing it.

### Commenting

Commenting is important. That's programmers are taught in the basic programming
lessons. However this does not mean that the more comments the better. Comments
should in the first place document what a function is doing. Not so much how it
does things. Once the reader knows what the code is supposed to do, well
written code explains on its own how it does it. On the other hand, if the
reader doesn't know what the code is supposed to do, even the most detailed
comments do not help.

Compare the following functions:

*Bad* example:
```python
def hypot(triangle):

    # reading in a
    a = triangle.get_a()

    # reading in b
    b = triangle.get_b()

    # reading in gamma
    gamma = triangle.get_gamma()

    # calculate c
    c = np.sqrt(a*a + b*b - 2*a*b*np.cos(gamma))

    # return result
    return c
```

Everyone sees that you read in some parameter `a`. Everyone sees that you read
in some parameter `b` and `gamma`. Everyone sees that you calculate and return
some value `c`. But what is it that you are doing?

Now the *good* example:
```python
def hypot(triangle):
    ''' Calculates the hypotenuse of a triangle using the law of cosines

    https://en.wikipedia.org/wiki/Law_of_cosines
    '''
    a = triangle.a()
    b = triangle.b()
    gamma = triangle.gamma()

    return np.sqrt(a*a + b*b - 2*a*b*np.cos(gamma))
```


[1]: https://www.python.org/dev/peps/pep-0008/
