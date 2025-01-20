# Contributing

Want to contribute? Great!


## Modus operandi

In principle we are using the standard GitHub pull request model. So in theory
you could simply file a pull request to us out of the blue and we merge it. In
practice that is not the ideal way of doing things. It is way better to get in
contact with us, before you start coding. Coordinating up front helps to avoid
frustration later on.

The natural way for that is to use the [issue
tracker](https://github.com/boschresearch/pylife/issues).  You can check what
we are currently working on and what is planned and discuss it with us in the
comments. Feel free to create your own issue if needed.

If you want to contribute something bigger, it is probably a good idea to first
start a discussion about it on the [discussion
page](https://github.com/boschresearch/pylife/discussions).  There we can
discuss if your contribution makes sense for pyLife and who can do what and how
long it will take. Probably we will setup a Kanban project or find some way to
split your contribution plan down to smaller pieces, that are better
reviewable.

## Test driven development

The functionality of your contribution (functions, class methods) needs to be
tested by [pytest][pytest] testing routines.  Please always run the test suite
locally before you file a pull request.  For any filed pull request the test
suite will be run for different python versions and different operating systems
as [GitHub action](https://github.com/boschresearch/pylife/actions).  If any of
them fails, please update your pull request accordingly.

Every bugfix or every new feature you are implementing – basically any function
changing code – must be backed with a suitable amount of unit tests. If those
unit tests are missing, we will have to decline your pull request.

In order to achieve maintainable code we ask contributors to use test driven
development, i. e. follow the [Three Rules of Test Driven
Development][rrugamba]:

1. Do not change production code without writing a failing unit test
   first. Cleanups and refactoring as well as performance optimizations are not
   changes in that sense.
2. Write only enough test code as is sufficient to fail.
3. Only write or change minimal production code as is sufficient to make the
   failing test pass.

We are measuring the testing coverage. Your pull request should not decrease
the test coverage although we are not enforcing that by our CI/CD. However, we
are manually checking during review if there are appropriate tests and we will
reject your contribution if they are missing and you are not adding them.

If you are modifying an existing test, you must deliver a very good reason for
it in your commit message. Otherwise we will probably reject your PR as being a
breaking change.

## Coding style

Please do consult the
[codingstyle](https://pylife.readthedocs.io/en/stable/CODINGSTYLE.html) file
for codingstyle guide lines. In order to have your contribution merged to main
line following guide lines should be met.

### Docstrings

Document your public API classes, methods, functions and attributes using numpy
style docstings unless the naming is *really* self-explanatory.

### Comments

Use as little comments as possible. The code along with docstrings should be
expressive enough. Remove any commented code lines before issuing your pull
request.

## Making commits

### Configure your git client

Please configure your identity in your git client appropriately. From the git
command line you can do that using

```sh
git config user.name <Your Name>
git config user.email <your-email@...>
```

### Writing good commit messages

Please consider following the [commit guidelines][CGL] when writing your commit
message. We will not enforce this, but we would appreciate if you
do. [Here][git-commit] is a good read why this makes sense.

## Branching and pull requests

Pull requests must be filed against the `develop` branch, except for urgent
bugfixes requiring a special bugfix release. Those can be filed against
`master`.

Branches should have meaningful names. Ideally there is an issue that the
branch addresses. Then the branch name should be prefixed with the number of
the issue.

Not that branches are meant to be short lived. That is why we don't have a
complex branch naming policy.

## License

Your contribution must be licensed under the Apache-2.0 license, the license
used by this project.

## Add / retain copyright notices

Include a copyright notice and license in each new file to be contributed,
consistent with the style used by this project. If your contribution contains
code under the copyright of a third party, document its origin, license, and
copyright holders.

## Sign your work

This project tracks patch provenance and licensing using the Developer
Certificate of Origin 1.1 (DCO) from [developercertificate.org][DCO] and
Signed-off-by tags initially developed by the Linux kernel project.

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

With the sign-off in a commit message you certify that you authored the patch
or otherwise have the right to submit it under an open source license. The
procedure is simple: To certify above Developer's Certificate of
Origin 1.1 for your contribution just append a line

```text
Signed-off-by: Random J Developer <random@developer.example.org>
```

to every commit message using your real name or your pseudonym and a valid
email address.

If you have set your `user.name` and `user.email` git configs you can
automatically sign the commit by running the git-commit command with the `-s`
option.  There may be multiple sign-offs if more than one developer was
involved in authoring the contribution.

Another option to automatically add the `Signed-off-by:` is to once use the
command

```sh
git config core.hooksPath .githooks
```

in your pyLife working directory. This will then add the `Signed-off-by:` line
automatically.

For a more detailed description of this procedure, please see
[SubmittingPatches][] which was extracted from the Linux kernel project, and
which is stored in an external repository.

Any pull request with at least one commit missing the Signed-off-tag in the
commit message will be automatically declined by a bot.

### Individual vs. Corporate Contributors

Often employers or academic institution have ownership over code that is
written in certain circumstances, so please do due diligence to ensure that
you have the right to submit the code.

If you are a developer who is authorized to contribute to pyLife on
behalf of your employer, then please use your corporate email address in the
Signed-off-by tag. Otherwise please use a personal email address.

## Maintain Copyright holder / Contributor list

Each contributor is responsible for identifying themselves in the
[NOTICE](NOTICE) file, the project's list of copyright holders and authors.
Please add the respective information corresponding to the Signed-off-by tag
as part of your first pull request.

If you are a developer who is authorized to contribute to pyLife on
behalf of your employer, then add your company / organization to the list of
copyright holders in the [NOTICE](NOTICE) file. As author of a corporate
contribution you can also add your name and corporate email address as in the
Signed-off-by tag.

If your contribution is covered by this project's DCO's clause "(c) The
contribution was provided directly to me by some other person who certified
(a) or (b) and I have not modified it", please add the appropriate copyright
holder(s) to the [NOTICE](NOTICE) file as part of your contribution.

[pytest]: https://pytest.org

[rrugamba]: https://medium.com/@rrugamba/3-laws-of-tdd-58b5ec46a998

[CGL]: https://www.git-scm.com/book/en/v2/Distributed-Git-Contributing-to-a-Project#_commit_guidelines

[git-commit]: https://chris.beams.io/posts/git-commit/

[DCO]: https://developercertificate.org/

[SubmittingPatches]: https://github.com/wking/signed-off-by/blob/7d71be37194df05c349157a2161c7534feaf86a4/Documentation/SubmittingPatches
