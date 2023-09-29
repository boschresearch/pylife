
pyLife coding style guidelines
==============================

Introduction
------------

One crucial quality criteria of program code is maintainability. In order to
maintain code, the code has to be written clearly so that it is easily readable
to someone who has not written it. Therefore it is helpful to have a consistent
coding style with consistent naming conventions. However, the coding style
rules are not supposed to be strict rules. They can be disobeyed if there are
good reasons to do so.

As we are programming in Python we vastly stick to the [PEP8 coding style
guide][1]. That document is generally recommendable to python programmers. This
document therefore covers only things that go beyond the PEP8. So please read
PEP8 for the general recommendations on python programming.

Clean code
^^^^^^^^^^

The notion of code quality that keeps software maintainable, makes it easier to
find and fix bugs and so on is nowadays referred to by the expression *Clean
Code*.

The iconic figure behind that notion is `Robert C. Martin <https://en.wikipedia.org/wiki/Robert_C._Martin>`_ aka Uncle Bob. For
the full story about clean code you can read his books *Clean Code* and *Clean
Coders*. Some of his lectures about Clean Code are available on Youtube.

Use a linter and let your editor help you
-----------------------------------------

A linter is a tool that scans your code and shows you where you are not
following the coding style guidelines. The anaconda environment of
``environment.yml`` comes with flake8 and pep8-naming, which warns about a lot of
things. Best is to configure your editor in a way that it shows you the linter
warnings as you type.

Many editors have some other useful helpers. For example whitespace cleanup,
i.e. delete any trailing whitespace as soon as you save the file.

Line lengths
------------

Lines should not often exceed the 90 characters. Exceeding it sometimes by a
bit is ok, though. Please do *never* exceed 125 characters because that's the
width of the GitHub code viewer.

Naming conventions
------------------

By naming conventions the programmer can give some indications to the reader of
the program, what an identifier is supposed to be or what it is referring
to. Therefore some consistency guidelines.

Mandatory names throughout the pyLife code base
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For variables representing physical quantities, we have a dedicated document in
the documentation.  Please follow the points discussed there.

Module names
^^^^^^^^^^^^

For module names, try to find one word names like ``rainflow``\ , ``gradient``. If
you by all means need word separation in a module name, use
``snake_case``. *Never* use dashes (\ ``-``\ ) and capital letters in module
names. They lead to all kinds of problems.

Class names
^^^^^^^^^^^

Class names are usually short and a single or compound noun. For these short
names we use the so called ``CamelCase`` style:

.. code-block:: python

   class DataObjectReader:
       ...

Function names
^^^^^^^^^^^^^^

Function and variable names can be longer than class names. Especially function
names tend to be actual sentences like:

.. code-block:: python

   def calc_all_data_from_scratch():
       ...

These are way more readable in the so called ``lowercase_with_underscores``
style.

Variable names
^^^^^^^^^^^^^^

Variable names can be shorter as long as they are local. For example when you
store the result of a function in a variable that the function is finally to
return, don't call it ``result_to_be_returned`` but only ``res``. A rule of thumb
is that the name of a variable needs to be descriptive, if the code part in
which the variable is used, exceeds the area that you can capture with one eye
glimpse.

Class method names
^^^^^^^^^^^^^^^^^^

There are a couple of conventions that make it easier to understand an API of a
class.

To access the data items of a class we used to use getter and setter
functions. A better and more modern way is python's ``@property`` decorator.

.. tip::

    Think twice before you implement setters.  Oftentimes you simply don't need
    them as you would usually simply create an object, use it and then throw it
    away without ever changing it.  So don't implement setters unless you
    *really* need them.  Even if you really need them it is often better to
    create a modified copy of a data structure rather than modifying the
    original.  Modifying objects can have side effects which can lead to subtle
    bugs.


.. code-block:: python

   class ExampleClass:
       def __init__(self):
           self._foo = 23
           self._bar = 42
           self._sum = None

       @property
       def foo(self):
           """ getter functions have the name of the accessed data item
           """
           return self._foo

       @foo.setter
       def foo(self, v):
           """ setter functions have the name of the accessed data item prefixed
               with `set_`
           """
           if v < 0: # sanity check
               raise Exception("Value for foo must be >= 0")
           self._foo = v

       def calc_sum_of_foo_and_bar(self):
           """ class methods whose name does not imply that they return data
               should not return anything.
           """
           self._sum = self._foo + self._bar

The old style getter and setter function like ``set_foo(self, new_foo)``\ are still
tolerable but should be avoided in new code. Before major releases we might dig
to the code and replace them with ``@property`` where feasible.

Structuring of the code
-----------------------

Data encapsulation
^^^^^^^^^^^^^^^^^^

One big advantage for object oriented programming is the so called data
encapsulation. That means that items of a class that is intended only for
internal use can be made inaccessible from outside of the class. Python does
not strictly enforce that concept, but in order to make it clear to the reader
of the code, we mark every class method and every class member variable that is
not meant to be accessed from outside the class with a leading underscore ``_``
like:

.. code-block:: python

   class Foo:

       def __init__(self):
           self.public_variable = 'bar'
           self._private_variable = 'baz'

       def public_method(self):
       ...

       def _private_method(self):

Object orientation
^^^^^^^^^^^^^^^^^^

Usually it makes sense to compound data structures and the functions using
these data structures into classes. The data structures then become class
members and the functions become class methods. This object oriented way of
doing things is recommendable but not always necessary. Sets of simple utility
routines can also be autonomous functions.

As a rule of thumb: If the user of some functionality needs to keep around a
data structure for a longer time and make several different function calls that
deal with the same data structure, it is probably a good idea to put everything
into a class.

Do not just put functions into a class because they belong semantically
together. That is what python modules are there for.

Functions and methods
^^^^^^^^^^^^^^^^^^^^^

Functions are not only there for sharing code but also to divide code into
easily manageable pieces. Therefore functions should be short and sweet and do
just one thing. If a function does not fit into your editor window, you should
consider to split it into smaller pieces. Even more so, if you need to scroll
in order to find out, where a loop or an if statement begins and ends. Ideally
a function should be as short, that it is no longer *possible* to extract a
piece of it.

Commenting
^^^^^^^^^^

Programmers are taught in the basic programming lessons that comments are
important. However, a more modern point of view is, that comments are only the
last resort, if the code is so obscure that the reader needs the comment to
understand it. Generally it would be better to write the code in a way that it
speaks for itself. That's why keeping functions short is so
important. Extracting a code block of a function into another function makes
the code more readable, because the new function has a name.

*Bad* example:

.. code-block:: python

   def some_function(data, parameters):
       ... # a bunch of code
       ... # over several lines
       ... # hard to figure out
       ... # what it is doing
       if parameters['use_method_1']:
           ... # a bunch of code
           ... # over several lines
           ... # hard to figure out
           ... # what it is doing
       else:
           ... # a bunch of code
           ... # over several lines
           ... # hard to figure out
           ... # what it is doing
       ... # a bunch of code
       ... # over several lines
       ... # hard to figure out
       ... # what it is doing

*Good* example

.. code-block:: python

   def prepare(data, parameters):
       ... # a bunch of code
       ... # over several lines
       ... # easily understandable
       ... # by the function's name

   def cleanup(data, parameters):
       ... # a bunch of code
       ... # over several lines
       ... # easily understandable
       ... # by the function's name

   def method_1(data):
       ... # a bunch of code
       ... # over several lines
       ... # easily understandable
       ... # by the function's name

   def other_method(data):
       ... # a bunch of code
       ... # over several lines
       ... # easily understandable
       ... # by the function's name

   def some_function(data, parameters):
       prepare(data, parameters)
       if parameters['use_method_1']:
           method_1(data)
       else:
           other_method(data)
       cleanup(data, parameters)

Ideally the only comments that you need are docstrings that document the public
interface of your functions and classes.

Compare the following functions:

*Bad* example:

.. code-block:: python

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

Everyone sees that you read in some parameter ``a``. Everyone sees that you read
in some parameter ``b`` and ``gamma``. Everyone sees that you calculate and return
some value ``c``. But what is it that you are doing?

Now the *good* example:

.. code-block:: python

   def hypot(triangle):
       """Calculate the hypotenuse of a triangle using the law of cosines

       https://en.wikipedia.org/wiki/Law_of_cosines
       """
       a = triangle.a
       b = triangle.b
       gamma = triangle.gamma

       return np.sqrt(a*a + b*b - 2*a*b*np.cos(gamma))
