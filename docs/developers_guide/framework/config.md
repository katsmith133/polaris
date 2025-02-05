(dev-config)=

# Config files

The primary documentation for the config parser is in
[MPAS-Tools config parser](http://mpas-dev.github.io/MPAS-Tools/stable/config.html).
Here, we include some specific details relevant to using the
{py:class}`mpas_tools.config.MpasConfigParser` in polaris.

Here, we provide the {py:class}`polaris.config.PolarisConfigParser` that has
almost the same functionality but also ensures that certain relative paths are
converted automatically to absolute paths.

The {py:meth}`mpas_tools.config.MpasConfigParser.add_from_package()` method can
be used to add the contents of a config file within a package to the config
options. Examples of this can be found in many test cases as well as
{py:func}`polaris.setup.setup_case()`. Here is a typical example from
{py:func}`polaris.ocean.tests.global_ocean.make_diagnostics_files.MakeDiagnosticsFiles.configure()`:

```python
def configure(self):
    """
    Modify the configuration options for this test case
    """
    self.config.add_from_package(
       'polaris.ocean.tests.global_ocean.make_diagnostics_files',
       'make_diagnostics_files.cfg', exception=True)
```

The first and second arguments are the name of a package containing the config
file and the name of the config file itself, respectively.  You can see that
the file is in the path `polaris/ocean/tests/global_ocean/make_diagnostics_files`
(replacing the `.` in the module name with `/`).  In this case, we know
that the config file should always exist, so we would like the code to raise
an exception (`exception=True`) if the file is not found.  This is the
default behavior.  In some cases, you would like the code to add the config
options if the config file exists and do nothing if it does not.  This can
be useful if a common configure function is being used for all test
cases in a configuration, as in this example from
{py:func}`setup.setup_case()`:

```python
# add the config options for the test group (if defined)
test_group = test_case.test_group.name
config.add_from_package(f'polaris.{component}.tests.{test_group}',
                        f'{test_group}.cfg', exception=False)
```

If a test group doesn't have any config options, nothing will happen.

The `MpasConfigParser` class also includes methods for adding a user
config file and other config files by file name, but these are largely intended
for use by the framework rather than individual test cases.

Other methods for the `MpasConfigParser` are similar to those for
{py:class}`configparser.ConfigParser`.  In addition to `get()`,
`getinteger()`, `getfloat()` and `getboolean()` methods, this class
implements {py:meth}`mpas_tools.config.MpasConfigParser.getlist()`, which
can be used to parse a config value separated by spaces and/or commas into
a list of strings, floats, integers, booleans, etc. Another useful method
is {py:meth}`mpas_tools.config.MpasConfigParser.getexpression()`, which can
be used to get python dictionaries, lists and tuples as well as a small set
of functions (`range()`, {py:meth}`numpy.linspace()`,
{py:meth}`numpy.arange()`, and {py:meth}`numpy.array()`)

## Comments in config files

One of the main advantages of {py:class}`mpas_tools.config.MpasConfigParser`
over {py:class}`configparser.ConfigParser` is that it keeps track of comments
that are associated with config sections and options.

See [comments in config files](http://mpas-dev.github.io/MPAS-Tools/stable/config.html#config_comments)
in MPAS-Tools for more details.
