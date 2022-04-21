import warnings
from pathlib import Path
import numpy as np
from typing import List

from pysd.translation.structures.abstract_model import AbstractSubscriptRange
from pysd.py_backend.external import ExtSubscript


class SubscriptManager:
    """
    SubscriptManager object allows saving the subscripts included in the
    Section, searching for elements or keys and simplifying them.

    Parameters
    ----------
    abstrac_subscripts: list
        List of the AbstractSubscriptRanges comming from the AbstractModel.

    _root: pathlib.Path
        Path to the model file. Needed to read subscript ranges from
        Excel files.

    """
    def __init__(self, abstract_subscripts: List[AbstractSubscriptRange],
                 _root: Path):
        self._root = _root
        self._copied = []
        self.mapping = {}
        self.subscripts = abstract_subscripts
        self.elements = {}
        self.subranges = self._get_main_subscripts()
        self.subscript2num = self._get_subscript2num()

    @property
    def subscripts(self) -> dict:
        return self._subscripts

    @subscripts.setter
    def subscripts(self, abstract_subscripts: List[AbstractSubscriptRange]):
        self._subscripts = {}
        missing = []
        for sub in abstract_subscripts:
            self.mapping[sub.name] = sub.mapping
            if isinstance(sub.subscripts, list):
                # regular definition of subscripts
                self._subscripts[sub.name] = sub.subscripts
            elif isinstance(sub.subscripts, str):
                # copied subscripts, this will be always a subrange,
                # then we need to prevent them of being saved as a main range
                self._copied.append(sub.name)
                self.mapping[sub.name].append(sub.subscripts)
                if sub.subscripts in self._subscripts:
                    self._subscripts[sub.name] =\
                        self._subscripts[sub.subscripts]
                else:
                    missing.append(sub)
            elif isinstance(sub.subscripts, dict):
                # subscript from file
                self._subscripts[sub.name] = ExtSubscript(
                    file_name=sub.subscripts["file"],
                    sheet=sub.subscripts["tab"],
                    firstcell=sub.subscripts["firstcell"],
                    lastcell=sub.subscripts["lastcell"],
                    prefix=sub.subscripts["prefix"],
                    root=self._root).subscript
            else:
                raise ValueError(
                    f"Invalid definition of subscript {sub.name}:\n\t"
                    + str(sub.subscripts))

        while missing:
            # second loop for copied subscripts
            sub = missing.pop()
            self._subscripts[sub.name] =\
                self._subscripts[sub.subscripts]

    def _get_main_subscripts(self) -> dict:
        """
        Reutrns a dictionary with the main ranges as keys and their
        subranges as values.
        """
        subscript_sets = {
            name: set(subs) for name, subs in self.subscripts.items()}

        subranges = {}
        for range, subs in subscript_sets.items():
            # current subscript range
            subranges[range] = []
            for subrange, subs2 in subscript_sets.items():
                if range == subrange:
                    # pass current range
                    continue
                elif subs == subs2:
                    # range is equal to the subrange, as Vensim does
                    # the main range will be the first one alphabetically
                    # make it case insensitive
                    range_l = range.replace(" ", "_").lower()
                    subrange_l = subrange.replace(" ", "_").lower()
                    if range_l < subrange_l and range not in self._copied:
                        subranges[range].append(subrange)
                    else:
                        # copied subscripts ranges or subscripts ranges
                        # that come later alphabetically
                        del subranges[range]
                        break
                elif subs2.issubset(subs):
                    # subrange is a subset of range, append it to the list
                    subranges[range].append(subrange)
                elif subs2.issuperset(subs):
                    # it exist a range that contents the elements of the range
                    del subranges[range]
                    break

        return subranges

    def _get_subscript2num(self) -> dict:
        """
        Build a dictionary to return the numeric value or values of a
        subscript or subscript range.
        """
        s2n = {}
        for range, subranges in self.subranges.items():
            # a main range is direct to return
            s2n[range.replace(" ", "_").lower()] = (
                f"np.arange(1, len(_subscript_dict['{range}'])+1)",
                {range: self.subscripts[range]}
            )
            for i, sub in enumerate(self.subscripts[range], start=1):
                # a subscript must return its numeric position
                # in the main range
                s2n[sub.replace(" ", "_").lower()] = (str(i), {})
            for subrange in subranges:
                # subranges may return the position of each subscript
                # in the main range
                sub_index = [
                    self.subscripts[range].index(sub)+1
                    for sub in self.subscripts[subrange]]

                if np.all(
                  sub_index
                  == np.arange(sub_index[0], sub_index[0]+len(sub_index))):
                    # subrange definition can be simplified with a range
                    subsarray = f"np.arange({sub_index[0]}, "\
                        f"len(_subscript_dict['{subrange}'])+{sub_index[0]})"
                else:
                    # subrange definition cannot be simplified
                    subsarray = f"np.array({sub_index})"

                s2n[subrange.replace(" ", "_").lower()] = (
                    subsarray,
                    {subrange: self.subscripts[subrange]}
                )

        return s2n

    def find_subscript_name(self, element: str, avoid: List[str] = []) -> str:
        """
        Given a member of a subscript family, return the first key of
        which the member is within the value list.
        If element is already a subscript name, return that.

        Parameters
        ----------
        element: str
            Subscript or subscriptrange name to find.
        avoid: list (optional)
            List of subscripts to avoid. Default is an empty list.

        Returns
        -------
        name: str
            The first key of which the member is within the value list
            in the subscripts dictionary.

        Examples
        --------
        >>> find_subscript_name('D')
        'Dim2'
        >>> find_subscript_name('B')
        'Dim1'
        >>> find_subscript_name('B', avoid=['Dim1'])
        'Dim2'

        """
        if element in self.subscripts.keys():
            return element

        for name, elements in self.subscripts.items():
            if element in elements and name not in avoid:
                return name

    def make_coord_dict(self, subs: List[str]) -> dict:
        """
        This is for assisting with the lookup of a particular element.

        Parameters
        ----------
        subs: list of strings
            Coordinates, either as names of dimensions, or positions within
            a dimension.

        Returns
        -------
        coordinates: dict
            Coordinates needed to access the xarray quantities we are
            interested in.

        Examples
        --------
        >>> make_coord_dict(['Dim1', 'D'])
        {'Dim1': ['A', 'B', 'C'], 'Dim2': ['D']}

        """
        sub_elems_list = [y for x in self.subscripts.values() for y in x]
        coordinates = {}
        for sub in subs:
            if sub in sub_elems_list:
                name = self.find_subscript_name(
                    sub, avoid=subs + list(coordinates))
                coordinates[name] = [sub]
            else:
                if sub.endswith("!"):
                    coordinates[sub] = self.subscripts[sub[:-1]]
                else:
                    coordinates[sub] = self.subscripts[sub]
        return coordinates

    def make_merge_list(self, subs_list: List[List[str]],
                        element: str = "") -> List[str]:
        """
        This is for assisting when building xrmerge. From a list of subscript
        lists returns the final subscript list after mergin. Necessary when
        merging variables with subscripts comming from different definitions.

        Parameters
        ----------
        subs_list: list of lists of strings
            Coordinates, either as names of dimensions, or positions within
            a dimension.
        element: str (optional)
            Element name, if given it will be printed with any error or
            warning message. Default is "".

        Returns
        -------
        dims: list
            Final subscripts after merging.

        Examples
        --------
        >>> sm = SubscriptManager()
        >>> sm.subscripts = {"upper": ["A", "B"], "all": ["A", "B", "C"]}
        >>> sm.make_merge_list([['upper'], ['C']])
        ['all']

        """
        coords_set = [set() for i in range(len(subs_list[0]))]
        coords_list = [
            self.make_coord_dict(subs)
            for subs in subs_list
        ]

        # update coords set
        [[coords_set[i].update(coords[dim]) for i, dim in enumerate(coords)]
         for coords in coords_list]

        dims = [None] * len(coords_set)
        # create an array with the name of the subranges for all
        # merging elements
        dims_list = np.array([
            list(coords) for coords in coords_list]).transpose()
        indexes = np.arange(len(dims))

        for i, coord2 in enumerate(coords_set):
            dims1 = [
                dim for dim in dims_list[i]
                if dim is not None and set(self.subscripts[dim]) == coord2
            ]
            if dims1:
                # if the given coordinate already matches return it
                dims[i] = dims1[0]
            else:
                # find a suitable coordinate
                other_dims = dims_list[indexes != i]
                for name, elements in self.subscripts.items():
                    if coord2 == set(elements) and name not in other_dims:
                        dims[i] = name
                        break

                if not dims[i]:
                    # the dimension is incomplete use the smaller
                    # dimension that completes it
                    for name, elements in self.subscripts.items():
                        if coord2.issubset(set(elements))\
                           and name not in other_dims:
                            dims[i] = name
                            warnings.warn(
                                element
                                + "\nDimension given by subscripts:"
                                + "\n\t{}\nis incomplete ".format(coord2)
                                + "using {} instead.".format(name)
                                + "\nSubscript_dict:"
                                + "\n\t{}".format(self.subscripts)
                            )
                            break

                if not dims[i]:
                    for name, elements in self.subscripts.items():
                        if coord2 == set(elements):
                            j = 1
                            while name + str(j) in self.subscripts.keys():
                                j += 1
                            self.subscripts[name + str(j)] = elements
                            dims[i] = name + str(j)
                            warnings.warn(
                                element
                                + "\nAdding new subscript range to"
                                + " subscript_dict:\n"
                                + name + str(j) + ": " + ', '.join(elements))
                            break

                if not dims[i]:
                    # not able to find the correct dimension
                    raise ValueError(
                        element
                        + "\nImpossible to find the dimension that contains:"
                        + "\n\t{}\nFor subscript_dict:".format(coord2)
                        + "\n\t{}".format(self.subscripts)
                    )

        return dims

    def simplify_subscript_input(self, coords: dict,
                                 merge_subs: List[str]) -> tuple:
        """
        Parameters
        ----------
        coords: dict
            Coordinates to write in the model file.

        merge_subs: list of strings
            List of the final subscript range of the python array after
            merging with other objects

        Returns
        -------
        final_subs, coords: dict, str
            Final subscripts and the equations to generate the coord
            dicttionary in the model file.

        """
        coordsp = []
        final_subs = {}
        for ndim, (dim, coord) in zip(merge_subs, coords.items()):
            # find dimensions can be retrieved from _subscript_dict
            final_subs[ndim] = coord
            if dim.endswith("!") and coord == self.subscripts[dim[:-1]]:
                # use _subscript_dict
                coordsp.append(f"'{ndim}': _subscript_dict['{dim[:-1]}']")
            elif not dim.endswith("!") and coord == self.subscripts[dim]:
                # use _subscript_dict
                coordsp.append(f"'{ndim}': _subscript_dict['{dim}']")
            else:
                # write whole dict
                coordsp.append(f"'{ndim}': {coord}")

        return final_subs, "{" + ", ".join(coordsp) + "}"
