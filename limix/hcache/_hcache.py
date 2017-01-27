import inspect
import os
import re
import six

try:
    xrange
except NameError:
    xrange = range

class Cached(object):

    def __init__(self):
        self._cache_groups = dict()
        self._diff_running = False

        regex = re.compile(r'^_cache_(.+)$')

        for (_, m) in inspect.getmembers(type(self),
                                         predicate=lambda p:
                                         (inspect.ismethod or
                                         inspect.isdatadescriptor)):

            if hasattr(m, 'fget'):
                f = m.fget
            elif inspect.ismethod(m):
                f = six.get_method_function(m)
            elif inspect.isfunction(m):
                f = m
            else:
                continue

            fv = six.get_function_code(f).co_freevars

            try:
                closure = six.get_function_closure(f)
            except AttributeError:
                continue

            if closure is None:
                continue

            vs = dict(zip(fv, (c.cell_contents for c in closure)))

            # this is used to make sure we are in the right function
            # i'm not proud of that, by the way
            if '_cache_identifier_pj97YCjgnp' not in vs:
                continue

            try:
                groups = vs['groups']
                method_name = re.match(regex, vs['cache_var_name']).group(1)
            except KeyError:
                continue

            for g in groups:
                if g not in self._cache_groups:
                    self._cache_groups[g] = []
                self._cache_groups[g].append(method_name)

            setattr(self, '_cache_' + method_name, None)
            setattr(self, '_cached_' + method_name, False)
            setattr(self, '_cached_args_' + method_name, dict())

    def _registered_methods(self):
        methods = set()
        for g in self._cache_groups.values():
            for m in g:
                methods.add(m)
        return list(methods)

    def clear_cache(self, *cache_groups):
        debug = int(os.getenv('HCACHE_DEBUG', 0))
        if debug:
            return
        assert hasattr(self, '_cache_groups'), "Cache system failed because "\
                                   + "you did not call Cached's constructor."

        for g in cache_groups:
            if g not in self._cache_groups:
                raise Exception('Cache group %s does not exist.' % g)

            for method_name in self._cache_groups[g]:
                setattr(self, '_cache_' + method_name, None)
                setattr(self, '_cached_' + method_name, False)
                setattr(self, '_cached_args_' + method_name, dict())

    def fill_cache(self, method_name, value):
        if method_name not in self._registered_methods():
            raise ValueError("The method %s is not registered in this cache."
                            % method_name)
        setattr(self, '_cache_' + method_name, value)
        setattr(self, '_cached_' + method_name, True)


# This decorator works both with or without arguments
# (i.e., @cached, @cached("group_name"), @cached(["group_name_A",
#       "group_name_B"]), or @cached(exclude=['param1', 'param2', ...]))
def cached(*args, **kwargs):
    deco_without_arg = len(args) == 1 and inspect.isfunction(args[0])

    # cached_args stores the argument names and argument values with which
    # the method was called in
    # cached_args = dict()
    filter_out = []
    _groups = ['default']
    # This is a hacky thing so that one can know that a method_wrapper
    # object, defined bellow, really came from here.
    _cache_identifier_pj97YCjgnp = [False]

    if not deco_without_arg:
        if len(args) > 0:
            if type(args[0]) is list or type(args[0]) is tuple:
                _groups += list(args[0])
            else:
                _groups.append(args[0])

        if len(kwargs) == 1:
            assert 'exclude' in kwargs, ("'exclude' is the only keyword "
                                         "allowed here.")
            filter_out = kwargs['exclude']
    else:
        _groups.append(args[0].__name__)

    defined_in_class = False
    frames = inspect.stack()
    if len(frames) > 2:
        f = frames[2][4]
        if f is None:
            try:
                frames[1][0].f_code.co_name
            except TypeError:
                pass
            else:
                defined_in_class = True
        else:
            defined_in_class =  f[0].strip().startswith('class ')

    def real_cached(method):
        if len(_groups) == 1 and _groups[0] == 'default':
            _groups.append(method.func_name)
        # dont look at this miserable hacky variable
        _cache_identifier_pj97YCjgnp[0] = True

        cache_var_name = '_cache_' + method.__name__
        valid_var_name = '_cached_' + method.__name__
        cached_args_name = '_cached_args_' + method.__name__
        groups = _groups

        debug = int(os.getenv('HCACHE_DEBUG', 0))
        if debug:
            return method

        def method_wrapper(self, *args, **kwargs):
            # dont look at this miserable hacky variable
            _cache_identifier_pj97YCjgnp[0] = True
            # this is meant to insert groups in this scope
            groups

            (argnames, argvalues) = _fetch_argnames_argvalues(method, args,
                                                              kwargs)

            t = zip(argnames, argvalues)
            provided_args = dict((x, y) for x, y in t)

            for f in filter_out:
                del provided_args[f]

            if getattr(self, valid_var_name, False) is False or\
                    provided_args != getattr(self, cached_args_name, dict()):

                if defined_in_class:
                    result = method(self, *args, **kwargs)
                else:
                    result = method(*args, **kwargs)

                setattr(self, cache_var_name, result)
                setattr(self, valid_var_name, True)
                if hasattr(self, cached_args_name):
                    getattr(self, cached_args_name).clear()
                    getattr(self, cached_args_name).update(provided_args)
                else:
                    setattr(self, cached_args_name, dict())

            return getattr(self, cache_var_name)

        if defined_in_class:
            return method_wrapper
        return lambda *args, **kwargs: method_wrapper(method, *args, **kwargs)

    if deco_without_arg:
        return real_cached(args[0])

    return real_cached


# ------------------------ INTERNAL USE ONLY ------------------------


def _map_args_kwargs_to_argvalues(args, kwargs, argnames, defaults):

    argvalues = [None] * len(argnames)
    for i in xrange(len(defaults)):
        argvalues[len(argnames) - len(defaults) + i] = defaults[i]

    for i in xrange(len(args)):
        argvalues[i] = args[i]

    for kw in kwargs:
        index = argnames.index(kw)
        argvalues[index] = kwargs[kw]

    return argvalues


# This function retrieves and organizes the method argument names and
# their values at the moment of the call. For this task,
# I need to both inspect the method (looking for default values) and
# the actual params passed by the user. Combining these two source of
# information I can return (argnames, argvalues) with all the argument names
# and argument values.
def _fetch_argnames_argvalues(method, args, kwargs):
    # argnames = inspect.getargspec(method)[0]
    nargs = six.get_function_code(method).co_argcount
    names = six.get_function_code(method).co_varnames
    argnames = list(names[:nargs])

    if len(argnames) == 0:
        return ([],[])

    if len(argnames) == 1 and argnames[0] == 'self':
        return ([],[])

    if argnames[0] == 'self':
        del argnames[0]

    defaults = six.get_function_defaults(method)
    if defaults is None:
        defaults = []
    argvalues = _map_args_kwargs_to_argvalues(args, kwargs, argnames, defaults)

    return (argnames, argvalues)
