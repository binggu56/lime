# encoding: utf-8
#
#Copyright (C) 2017-2020, P. R. Wiecha
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Collection of 2D visualization tools for pyGDM2

all 2D plotting is done using `matplotlib`

"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import copy

from . import tools

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import warnings



def _automatic_projection_select(X,Y,Z):
    if  len(np.unique(Z))==1:
        projection = 'xy'
    elif  len(np.unique(Y))==1:
        projection = 'xz'
    elif  len(np.unique(X))==1:
        projection = 'yz'
    else:
        warnings.warn("3D data. Falling back to XY projection...")
        projection = 'xy'   # fallback

    return projection






########################################################################
##                      VISUALIZATION FUNCTIONS
########################################################################
def structure(struct, projection='auto', color='auto', scale=1,
                 borders=3, marker='s', lw=0.1, EACHN=1, tit='',
                 plot_legend=True,
                 ax=None, absscale=False, show=True, **kwargs):
    """plot structure in 2d, projected to e.g. the XY plane

    plot the structure `struct` as a scatter projection to a carthesian plane.
    Either from list of coordinates, or using a simulation definition as input.

    kwargs are passed to matplotlib's `scatter`

    Parameters
    ----------
    struct : list or :class:`.core.simulation`
          either list of 3d coordinate tuples or simulation description object

    projection : str, default: 'auto'
        which 2D projection to plot: "auto", "XY", "YZ", "XZ"

    color : str or matplotlib color, default: "auto"
            Color of scatterplot. Either "auto", or matplotlib-compatible color.
            "auto": automatic color selection (multi-color for multiple materials).

    scale : float, default: 0.5
          symbol scaling in units of stepsize

    borders : float, default: 3
          additional space limits around plot in units of stepsize

    marker : str, default "s" (=squares)
          scatter symbols for plotting meshpoints. Convention of
          matplotlib's `scatter`

    lw : float, default 0.1
          line width around scatter symbols. Convention of matplotlib's `scatter`

    EACHN : int, default: 1 (=all)
          show each `EACHN` points only

    tit : str, default: ""
        title for plot (optional)

    plot_legend : bool, default: True
        whether to add a legend if multi-material structure (requires auto-color enabled)

    ax : matplotlib `axes`, default: None (=create new)
           axes object (matplotlib) to plot into

    absscale : bool, default: False
          absolute or relative scaling. If True, override internal
          scaling calculation

    show : bool, default: True
          directly show plot

    Returns
    -------
    result returned by matplotlib's `scatter`

    """
    from . import structures

    X,Y,Z = tools.get_geometry(struct)
    step = tools.get_step_from_geometry(struct)

    if projection.lower() == 'auto':
        projection = _automatic_projection_select(X,Y,Z)

    if projection.lower() == 'xy':
        X=X; Y=Y
    elif projection.lower() == 'yz':
        X=Y; Y=Z
    elif projection.lower() == 'xz':
        X=X; Y=Z
    else:
        raise ValueError("Invalid projection parameter!")

    ## -- colors for subsets with different materials
    if color == 'auto':
        if hasattr(struct, "struct") or type(struct) == structures.struct:
            if hasattr(struct, "struct"):
                struct = struct.struct

            if hasattr(struct.material, '__iter__'):
                materials = [s.__name__ for s in struct.material]
                if len(set(materials)) > 1:
                    a = np.ascontiguousarray(np.transpose([X,Y]))
                    materials_idx = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=1)[1]
                    material = np.array(materials)[materials_idx]
                    different_materials = np.unique(materials)
                    indices_subsets = []
                    for struct_fraction in different_materials:
                        indices_subsets.append(np.arange(len(material))[material==struct_fraction])
                else:
                    color = '.2'
            else:
                color = '.2'
        else:
            color = '.2'

    ## -- del all duplicates with same x/y coordinates (for plot efficiency)
    X,Y = tools.unique_rows(np.transpose([X,Y]))[::EACHN].T

    if not ax:
        ax = plt.gca()

    if show:
        plt.gca().set_aspect('equal')
        plt.xlim(X.min()-borders*step, X.max()+borders*step)
        plt.ylim(Y.min()-borders*step, Y.max()+borders*step)

    ## -- set scaling depending on structure size
    if not absscale:
        bbox = plt.gca().get_window_extent().transformed(plt.gcf().dpi_scale_trans.inverted())
        axes_pixels_w = bbox.width * plt.gcf().dpi
        axes_pixels_h = bbox.height * plt.gcf().dpi
        maxdist_w = max(X) - min(X) + step
        maxdist_h = max(Y) - min(Y) + step
        N_dp_max_w = maxdist_w / step
        N_dp_max_h = maxdist_h / step

        scale_w = scale/1.2 * axes_pixels_w * maxdist_w/((maxdist_w + 2*step*borders) * N_dp_max_w)
        scale_h = scale/1.2 * axes_pixels_h * maxdist_h/((maxdist_h + 2*step*borders) * N_dp_max_h)

        scale = min([scale_w, scale_h])

    ## --- if multi-material: set maker colors
    if color != 'auto':
        im = ax.scatter(X,Y, marker=marker, s=scale**2,
                        c=color, edgecolors=color, lw=lw, **kwargs)
    else:
        colors = ['C{}'.format(i) for i in range(1,10)]*int(2 + len(indices_subsets)/9.)
        for i, idx in enumerate(indices_subsets):
            col = colors[i]
            if plot_legend:
                label = different_materials[i]
            else:
                label = ''
            im = ax.scatter(X[idx],Y[idx], marker=marker, s=scale**2,
                            c=col, edgecolors=col, label=label, lw=lw, **kwargs)
        if plot_legend:
            plt.legend()

    if tit:
        plt.title(tit)

    if show:
        plt.xlabel("{} (nm)".format(projection[0]))
        plt.ylabel("{} (nm)".format(projection[1]))
        plt.show()

    return im



def structure_contour(struct, projection='auto', color='b',
                      N_points=4, borders=3, lw=1, s=5, style='lines',
                      input_mesh='cube', tit='', ax=None,
                      add_first_point=True, show=True, **kwargs):
    """Contour around structure

    further kwargs are passed to matplotlib's  `scatter` or `plot`

    Parameters
    ----------
    struct : list or :class:`.core.simulation`
          either list of 3d coordinate tuples or simulation description object

    projection : str, default: 'auto'
        which 2D projection to plot: "auto", "XY", "YZ", "XZ"

    color : str or matplotlib color, default: "b"
          matplotlib-compatible color for scatter

    N_points : int, default: 4
          number of additional points to calculate at each meshpoint-side
          (basically meant for style "dots", default: 4)

    borders : float, default: 3
          additional space limits around plot in units of stepsize

    lw : float, default: 1
          linewidth as used by `pyplot.plot` (style='lines' only)

    s : float, default: 5
          symbol scaling for scatterplot as used by `pyplot.scatter`
          (style='dots' only)

    stlye : str, default: 'lines'
          Style of plot. One of ["dots", "lines"]

    input_mesh : str, default: 'cube'
          mesh type, default: "cube". one of ["cube", "hex", "hex2", "hex_onelayer"]

    tit : str, default: ""
        title for plot (optional)

    ax : matplotlib `axes`, default: None (=create new)
           axes object (matplotlib) to plot into

    add_first_point : bool, default: True
          If True, adds first point twice in order to close the contour.
          Might cause some strange lines if a single edge meshpoint is first in
          the geometry.

    show : bool, default: True
          directly show plot

    Returns
    -------
    matplotlib's `scatter` or `line` return value :
        contour plot object

    """
    if ax is None:
        ax = plt.gca()

    if N_points<2:
        raise ValueError("'N_points' must be >= 2!")

    geo = tools.get_geometry(struct)
    if projection.lower() == 'auto':
        projection = _automatic_projection_select(*geo)

    ## test if only one layer
    if input_mesh.lower() in ['hex', 'hex1', 'hex2']:
        if projection.lower() == 'xy':
            if len(np.unique(geo[2])) == 1: input_mesh='hex_onelayer'
        if projection.lower() == 'xz':
            if len(np.unique(geo[1])) == 1: input_mesh='hex_onelayer'
        if projection.lower() == 'yz':
            if len(np.unique(geo[0])) == 1: input_mesh='hex_onelayer'

    struct2D = tools.get_geometry_2d_projection(struct, projection=projection)
    struct2D = np.round(struct2D, 3)


    from . import core
    if type(struct) == core.simulation:
        if struct.struct.normalization != 1.0 and input_mesh.lower() == 'cube':
            warnings.warn("Simulation seems hexagonal mesh, but `input_mesh` parameter is set to 'cube'.")
        if struct.struct.normalization == 1.0 and input_mesh.lower() in ['hex', 'hex1', 'hex2']:
            warnings.warn("Simulation seems cubic mesh, but `input_mesh` parameter is set to hexagonal.")

    if input_mesh.lower() == 'cube':
        NN_bulk, mb = 4, 1.1
        NN_sf = 4
    elif input_mesh.lower() in ['hex', 'hex1']:
        NN_bulk, mb = 3, np.sqrt(2)
        NN_sf = 3
    elif input_mesh.lower() == 'hex2':
        NN_bulk, mb = 4, np.sqrt(1.3)
        NN_sf = 4
    elif input_mesh.lower() == 'hex_onelayer':
        NN_bulk, mb = 6, np.sqrt(2)
        NN_sf = 6
    SF, SF_vec = tools.get_surface_meshpoints(struct2D, NN_bulk=NN_bulk, max_bound=mb,
                                              NN_surface=NN_sf, max_bound_sf=5.0)

    step = tools.get_step_from_geometry(struct)
    d = step/2.
    if input_mesh.lower() in ['hex', 'hex1', 'hex2']:
        d /= 2.#np.sqrt(2)
    if input_mesh.lower() == 'hex_onelayer':
        d /= 2.

    if projection.lower() == 'xy':
        coord_idx = [0,1]
    elif projection.lower() == 'yz':
        coord_idx = [1,2]
    elif projection.lower() == 'xz':
        coord_idx = [0,2]

    ## --- calculate the contour-pixel positions
    SF_plot = []
    for i, vec in enumerate(SF_vec):
        x, y = SF[i][coord_idx]
#==============================================================================
#         CUBE MESH
#==============================================================================
        if input_mesh.lower() == 'cube':
            if round(vec[coord_idx[0]]) != 0:
                for i in range(N_points):
                    SF_plot.append([x + np.sign(vec[coord_idx[0]])*d,
                                    y-d + d*i/((N_points-1)/2.)])

            if round(vec[coord_idx[1]]) != 0:
                    for i in range(N_points):
                        SF_plot.append([x-d + d*i/((N_points-1)/2.),
                                        y + np.sign(vec[coord_idx[1]])*d])
#==============================================================================
#         HEXAGONAL MESH
#==============================================================================
        if input_mesh.lower() in ['hex', 'hex1', 'hex2', 'hex_onelayer']:
            SF_plot.append([x + d*vec[coord_idx[0]],
                            y + d*vec[coord_idx[1]]])
    if len(SF_plot)==0:
        raise ValueError("Empty surface data! Please check mesh-type (parameter `input_mesh`) and input structure data.")
    X, Y = tools.unique_rows(np.array(SF_plot)).T

    ## --- plot as scatter
    if style.lower() == 'dots':
        cont = ax.scatter(X, Y, color=color, s=s, **kwargs)

    ## --- plot as lines
    elif style.lower() == 'lines':
        from scipy.spatial import cKDTree as KDTree
        geometry_line = np.transpose([X, Y])

        ## sort coordinates such, that each point follows a closest neighbor
        i_pos = 0
        ordered_list = [ [geometry_line[i_pos]] ]
        for i in range(len(geometry_line)-1):
            pos = geometry_line[i_pos]

            ## query for nearest neighbor (exclude "self-position")
            kdtree = KDTree(geometry_line)
            closest = kdtree.query(pos, k=2)
            idx_closest = closest[1][1]

            ## stop if no next neighbour found
            if idx_closest >= len(geometry_line):
                break

            ## if separate structure detected: start new list for contour-line
            min_dist_fact = 0.9
            if input_mesh.lower() in ['hex', 'hex1', 'hex2', 'hex_onelayer']:
                min_dist_fact = 3.0
            if closest[0][1] > step*1.4*min_dist_fact:
                ordered_list[-1].append(ordered_list[-1][0])
                ordered_list.append([])

            ordered_list[-1].append(geometry_line[idx_closest])
            geometry_line = np.delete(geometry_line, i_pos, 0)

            if i_pos < idx_closest:
                idx_closest -= 1
            i_pos = idx_closest

        ordered_list[-1].append(geometry_line[0])  # add last remaining point
        if add_first_point:
            ordered_list[-1].append(ordered_list[-1][0])  # add first point to close the loop

        ## --- plot all contours
        for sub_struct in ordered_list:
            Xs, Ys = np.transpose(sub_struct)
            cont = ax.plot(Xs,Ys, color=color, lw=lw, **kwargs)

    else:
        raise ValueError("`style` must be one of ['dots', 'lines']")

    if tit:
        ax.set_title(tit)

    if show:
        ax.set_aspect("equal")
        ax.set_xlabel("{} (nm)".format(projection[0]))
        ax.set_ylabel("{} (nm)".format(projection[1]))
        ax.set_xlim(X.min()-borders*step, X.max()+borders*step)
        ax.set_ylim(Y.min()-borders*step, Y.max()+borders*step)

        plt.show()

    return cont




##----------------------------------------------------------------------
##                     INTERNAL FIELD - FUNDAMENTAL
##----------------------------------------------------------------------
def vectorfield(NF, struct=None, projection='auto', complex_part='real',
                tit='',
                slice_level=None,
                scale=10.0, vecwidth=1.0, cmap=cm.Blues, cmin=0.3,
                ax=None, adjust_axis=True, show=True,
                borders=3, EACHN=1, sortL=True, overrideMaxL=False,
                **kwargs):
    """plot 2D Vector field as quiver plot

    plot nearfield list as 2D vector plot, using matplotlib's `quiver`.
    `kwargs` are passed to `pyplot.quiver`

    Parameters
    ----------
    NF : list of 3- or 6-tuples
        Nearfield definition. `np.array`, containing 6-tuples:
        (X,Y,Z, Ex,Ey,Ez), the field components being complex (use e.g.
        :func:`.tools.get_field_as_list`).
        Optionally can also take a list of 3-tuples (Ex, Ey, Ez),
        in which case the structure must be provided via the `struct` kwarg.

    struct : list or :class:`.core.simulation`, optional
        optional structure definition (necessary if field is supplied in
        3-tuple form without coordinates). Either `simulation` description
        object, or list of (x,y,z) coordinate tuples

    projection : str, default: 'auto'
        Which 2D projection to plot: "auto", "XY", "YZ", "XZ"

    complex_part : str, default: 'real'
        Which part of complex field to plot. Either 'real' or 'imag'.

    tit : str, default: ""
        title for plot (optional)

    slice_level: float, default: `None`
        optional value of depth where to slice. eg if projection=='XY',
        slice_level=10 will take only values where Z==10.
            - slice_level = `None`, plot all vectors one above another without slicing.
            - slice_level = -9999 : take minimum value in field-list.
            - slice_level = 9999 : take maximum value in field-list.

    scale : float, default: 10.0
        optional vector length scaling parameter

    vecwidth : float, default: 1.0
        optional vector width scaling parameter

    cmap : matplotlib colormap, default: `cm.Blues`
        matplotlib colormap to use for arrows (color scaling by vector length)

    cmin : float, default: 0.3
        minimal color to use from cmap to avoid pure white

    ax : matplotlib `axes`, default: None (=create new)
        optinal axes object (mpl) to plot into

    adjust_axis : bool, default: True
        apply adjustments to axis (equal aspect, axis labels, plot-range)

    show : bool, default: True
        whether or not to call `pyplot.show`

    borders : float, default: 3
          additional space limits around plot in units of stepsize

    EACHN : int, default: 1 [=all]
        show each N points only

    sortL : bool, default: True
        sort vectors by length to avoid clipping (True: Plot longest
        vectors on top)

    Returns
    -------

    return value of matplotlib's `quiver`

    """
    NF_plt = copy.deepcopy(NF)
    if len(NF) == 2:
        NF_plt = NF_plt[1]

    if len(NF_plt.T) == 6:
        X,Y,Z, UXcplx,UYcplx,UZcplx = np.transpose(NF_plt)
    elif len(NF_plt.T) == 3 and struct is not None:
        UXcplx,UYcplx,UZcplx = np.transpose(NF_plt)
        X,Y,Z = tools.get_geometry(struct)
    else:
        raise ValueError("Error: Wrong number of columns in vector field. Expected (Ex,Ey,Ez)-tuples + `simulation` object or (x,y,z, Ex,Ey,Ez)-tuples.")

    step = tools.get_step_from_geometry(np.transpose([X,Y,Z]))

    if projection.lower() == 'auto':
        projection = _automatic_projection_select(X,Y,Z)

    if complex_part.lower() == "real":
        UX, UY, UZ = UXcplx.real, UYcplx.real, UZcplx.real
    elif complex_part.lower() == "imag":
        UX, UY, UZ = UXcplx.imag, UYcplx.imag, UZcplx.imag
    else:
        raise ValueError("Error: Unknown `complex_part` argument value. Must be either 'real' or 'imag'.")

    if projection.lower() == 'xy':
        SLICE = copy.deepcopy(Z)
        X=X[::EACHN].real; Y=Y[::EACHN].real
        UX=UX[::EACHN]; UY=UY[::EACHN]
    elif projection.lower() == 'yz':
        SLICE = copy.deepcopy(X)
        X=Y[::EACHN].real; Y=Z[::EACHN].real
        UX=UY[::EACHN]; UY=UZ[::EACHN]
    elif projection.lower() == 'xz':
        SLICE = copy.deepcopy(Y)
        X=X[::EACHN].real; Y=Z[::EACHN].real
        UX=UX[::EACHN]; UY=UZ[::EACHN]
    else:
        raise ValueError("Invalid projection parameter!")


    ## -- optional slicing
    if slice_level is not None:
        if slice_level == -9999: slice_level = np.min(SLICE)
        if slice_level == 9999: slice_level = np.max(SLICE)
        X, Y   = X[(SLICE == slice_level)], Y[(SLICE == slice_level)]
        UX, UY = UX[(SLICE == slice_level)], UY[(SLICE == slice_level)]


    ## -- sort and scale by length
    def sortbylength(X,Y,EX,EY):
        ## -- sort by vector length to plot short vectors above long ones
        DATARR = []
        for i, (xi, yi, pxi, pyi) in enumerate(zip(X,Y,EX,EY)):
            DATARR.append([(pxi**2+pyi**2),xi,yi,pxi,pyi])

        DATARR = np.transpose(sorted(DATARR))

        X = DATARR[1]
        Y = DATARR[2]
        UX = DATARR[3]
        UY = DATARR[4]

        return X,Y,UX,UY
    if sortL:
        X,Y,UX,UY = sortbylength(X,Y,UX,UY)

    LenALL = np.sqrt(UX**2 + UY**2)
    if not overrideMaxL:
        scale = scale * LenALL.max()
    else:
        scale = scale

    ## -- quiver plot
    vecwidth = vecwidth * 0.005
    cscale = mcolors.Normalize( LenALL.min()-cmin*(LenALL.max()-LenALL.min()),
                                LenALL.max())   # cmap colors by vector length
    if ax != None:
        if tit:
            ax.set_title(tit)
        im = ax.quiver(X,Y, UX,UY, scale=scale,
                       width=vecwidth, color=cmap(cscale(LenALL)), **kwargs)
    else:
        if tit:
            plt.title(tit)
        im = plt.quiver(X,Y, UX,UY, scale=scale,
                        width=vecwidth, color=cmap(cscale(LenALL)), **kwargs)
        ax = plt.gca()

    if adjust_axis:
        ax.set_xlim(X.min() - borders*step, X.max() + borders*step)
        ax.set_ylim(Y.min() - borders*step, Y.max() + borders*step)
        ax.set_aspect("equal")
        ax.set_xlabel("{} (nm)".format(projection[0]))
        ax.set_ylabel("{} (nm)".format(projection[1]))

    if show:
        plt.show()

    return im


def vectorfield_by_fieldindex(sim, field_index, **kwargs):
    """Wrapper to :func:`.vectorfield`, using simulation object and fieldindex as input

    All other keyword arguments are passed to :func:`.vectorfield`.

    Parameters
    ----------
    sim : `simulation`
        instance of :class:`.core.simulation`

    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`

    """
    NF = tools.get_field_as_list_by_fieldindex(sim, field_index)
    return vectorfield(NF, **kwargs)






def vectorfield_fieldlines(NF, projection='auto', tit='',
                           complex_part='real', borders=0,
                           NX=-1, NY=-1, show=True, **kwargs):
    """2d plot of fieldlines of field 'NF'

    other optional arguments passed to matplotlib's `pyplot.streamplot`

    Parameters
    ----------
    NF : list of 6-tuples
        Nearfield definition. `np.array`, containing 6-tuples:
        (X,Y,Z, Ex,Ey,Ez), the field components being complex (use e.g.
        :func:`.tools.get_field_as_list`).

    projection : str, default: 'XY'
        Which projection to plot: "XY", "YZ", "XZ"

    complex_part : str, default: 'real'
        Which part of complex field to plot. Either 'real' or 'imag'.

    tit : str, default: ""
        title for plot (optional)

    borders : float, default: 0
          additional space limits around plot in units of stepsize

    NX, NY : int, int, defaults: -1, -1
        optional interpolation steps for nearfield.
        by default take number of independent positions in NF
        (if NX or NY == -1)

    show : bool, default: True
        whether or not to call `pyplot.show`


    Returns
    -------
        if show == False :
            return matplotlib `streamplot` object
    """
    if len(NF.T) == 6:
        x,y,z, Ex,Ey,Ez = NF.T
        x = x.real
        y = y.real
        z = z.real
    else:
        raise ValueError("Error: Field list must contain tuples of exactly 6 elements.")

    step = tools.get_step_from_geometry(np.transpose([x,y,z]))

    if projection.lower() == 'auto':
        projection = _automatic_projection_select(x, y, z)

    if complex_part.lower() == "real":
        Ex, Ey, Ez = Ex.real, Ey.real, Ez.real
    elif complex_part.lower() == "imag":
        Ex, Ey, Ez = Ex.imag, Ey.imag, Ez.imag
    else:
        raise ValueError("Error: Unknown `complex_part` argument value. " +
                         "Must be either 'real' or 'imag'.")


    if projection.lower() == 'xy':
        X=x; Y=y; EX=Ex; EY=Ey
    if projection.lower() == 'xz':
        X=x; Y=z; EX=Ex; EY=Ez
    if projection.lower() == 'yz':
        X=y; Y=z; EX=Ey; EY=Ez

    X = np.unique(X)
    Y = np.unique(Y)

    if NX == -1:
        NX = len(X)
    if NY == -1:
        NY = len(Y)

    MAP_XY = tools.generate_NF_map(np.min(X),np.max(X),NX, np.min(Y),np.max(Y),NY)

    NF_X, _extent = tools.map_to_grid_2D(MAP_XY, EX)
    NF_Y, _extent = tools.map_to_grid_2D(MAP_XY, EY)

    strplt = plt.streamplot(X, Y, NF_X, NF_Y, **kwargs)

    plt.xlim(_extent[0]-borders*step, _extent[1]+borders*step)
    plt.ylim(_extent[2]-borders*step, _extent[3]+borders*step)

    if show:
        if tit:
            plt.title(tit)
        plt.gca().set_aspect("equal")
        plt.xlabel("{} (nm)".format(projection[0]))
        plt.ylabel("{} (nm)".format(projection[1]))
        plt.show()
    else:
        return strplt



## 2D colorplot
def vectorfield_color(NF, projection='auto', complex_part='real',
                      tit='', ax=None,
                      slice_level=-9999, NX=None, NY=None, fieldComp='I',
                      borders=0,
                      cmap='seismic', clim=None, show=True,
                      **kwargs):
    """plot of 2D field data as colorplot

    other kwargs are passed to matplotlib's `imshow`

    Parameters
    ----------
    NF : list of 6-tuples
        Nearfield definition. `np.array`, containing 6-tuples:
        (X,Y,Z, Ex,Ey,Ez), the field components being complex (use e.g.
        :func:`.tools.get_field_as_list`).

    projection : str, default: 'auto'
        which 2D projection to plot: "auto", "XY", "YZ", "XZ"

    complex_part : str, default: 'real'
        Which part of complex field to plot. Either 'real' or 'imag'.

    tit : str, default: ""
        title for plot (optional)

    ax : matplotlib `axes`, default: None (=create new)
        optinal axes object (mpl) to plot into

    slice_level: float, default: `None`
        optional value of depth where to slice. eg if projection=='XY',
        slice_level=10 will take only values where Z==10.
            - slice_level = `None`, plot all vectors one above another without slicing.
            - slice_level = -9999 : take minimum value in field-list.
            - slice_level = 9999 : take maximum value in field-list.

    NX, NY : int, int, defaults: None, None
        optional interpolation steps for nearfield.
        by default take number of independent positions in NF
        (if NX or NY == None)

    fieldComp : str, default: 'I'
        Which component to use. One of ["I", "Ex", "Ey", "Ez"].
        If "I" is used, `complex_part` argument has no effect.

    borders : float, default: 0
        additional space in nm to plotting borders

    cmap : matplotlib colormap, default: "seismic"
        matplotlib colormap to use for colorplot

    clim : float, default: None
        optional colormap limits to pass to `plt.clim()`

    show : bool, default: True
        whether or not to call `pyplot.show`


    Returns
    -------
    result of matplotlib's `imshow`
    """
    if ax is None:
        ax = plt.gca()

    if len(NF.T) == 6:
        X,Y,Z, Ex,Ey,Ez = NF.T
        X = X.real
        Y = Y.real
        Z = Z.real
    else:
        raise ValueError("Error: Field list must contain tuples of exactly 6 elements.")

    step = tools.get_step_from_geometry(np.transpose([X,Y,Z]))

    if projection.lower() == 'auto':
        projection = _automatic_projection_select(X,Y,Z)

    if fieldComp.lower() != 'i':
        if complex_part.lower() == "real":
            Ex, Ey, Ez = Ex.real, Ey.real, Ez.real
        elif complex_part.lower() == "imag":
            Ex, Ey, Ez = Ex.imag, Ey.imag, Ez.imag
        else:
            raise ValueError("Error: Unknown `complex_part` argument value." +
                             " Must be either 'real' or 'imag'.")


    if fieldComp.lower() == 'i':
        EF = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    elif fieldComp.lower() == 'ex':
        EF = Ex
    elif fieldComp.lower() == 'ey':
        EF = Ey
    elif fieldComp.lower() == 'ez':
        EF = Ez

    if projection.lower() == 'xy':
        if slice_level == -9999: slice_level = np.min(Z)
        if slice_level == 9999: slice_level = np.max(Z)
        XYZList = np.transpose([X[(Z == slice_level)],Y[(Z == slice_level)], EF[(Z == slice_level)]])
    if projection.lower() == 'xz':
        if slice_level == -9999: slice_level = np.min(Y)
        if slice_level == 9999: slice_level = np.max(Y)
        XYZList = np.transpose([X[(Y == slice_level)],Z[(Y == slice_level)], EF[(Y == slice_level)]])
    if projection.lower() == 'yz':
        if slice_level == -9999: slice_level = np.min(X)
        if slice_level == 9999: slice_level = np.max(X)
        XYZList = np.transpose([Y[(X == slice_level)],Z[(X == slice_level)], EF[(X == slice_level)]])

    MAP, extent = tools.list_to_grid(XYZList, NX, NY, interpolation='cubic')

    if tit: ax.set_title(tit)
    img = ax.imshow(MAP, extent=extent, cmap=cmap, **kwargs)
    if clim: ax.set_clim(clim)

    ax.set_xlim(extent[0] - borders*step, extent[1] + borders*step)
    ax.set_ylim(extent[2] - borders*step, extent[3] + borders*step)

    if show:
        if fieldComp.lower() == 'i':
            plt.colorbar(img, label=r'$|E|^2 / |E_0|^2$')
        else:
            plt.colorbar(img, label=r'$E / |E_0|$')
        ax.set_aspect("equal")
        ax.set_xlabel("{} (nm)".format(projection[0]))
        ax.set_ylabel("{} (nm)".format(projection[1]))
        plt.show()
    else:
        return img


def vectorfield_color_by_fieldindex(sim, field_index, **kwargs):
    """Wrapper to :func:`.vectorfield_color`, using simulation object and fieldindex as input

    All other keyword arguments are passed to :func:`.vectorfield_color`.

    Parameters
    ----------
    sim : `simulation`
        instance of :class:`.core.simulation`

    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`

    """
#    x, y, z = sim.struct.geometry.T
#    Ex, Ey, Ez = sim.E[field_index][1].T
#    NF = np.transpose([x,y,z, Ex,Ey,Ez])
    NF = tools.get_field_as_list_by_fieldindex(sim, field_index)

    return vectorfield_color(NF, **kwargs)


def scalarfield(NF, **kwargs):
    """Wrapper to :func:`.vectorfield_color`, using scalar data tuples (x,y,z,S) as input

    All other keyword arguments are passed to :func:`.vectorfield_color`.

    Parameters
    ----------
    NF : list of 4-tuples
        list of tuples (x,y,z,S).
        Alternatively, the scalar-field can be passed as list of 2 lists
        containing the x/y positions and scalar values, respectively.
        ([xy-values, S], with xy-values: list of 2-tuples [x,y]; S: list of
        scalars). This format is returned e.g. by
        :func:`.tools.calculate_rasterscan`.

    """
    if len(NF) == 2 and np.shape(NF[0])[1] == 2 and len(np.shape(NF[1])) == 1 \
                    and len(NF[0]) == len(NF[1]):
        NF = np.concatenate([NF[0].T, [np.zeros(len(NF[1]))], [NF[1]]]).T
    elif len(NF.T) != 4:
        NF = np.array(NF).T
        if len(NF.T) != 4:
            raise ValueError("Error: Scalar field must consist of 4-tuples (x,y,z,S).")
    NF = np.array(NF)

    x, y, z = NF.T[0:3]
    Ex = Ey = Ez = NF.T[3]
    NF = np.transpose([x,y,z, Ex,Ey,Ez])

    im = vectorfield_color(NF, complex_part='real', fieldComp='Ex', **kwargs)
    return im








##----------------------------------------------------------------------
##               FARFIELD (INTENSITY)
##----------------------------------------------------------------------
def farfield_pattern_2D(theta, phi, I, degrees=True,
                        show=True, ax=None, **kwargs):
    """Plot BFP-like 2D far-field radiation pattern

    Plot a "back-focal plane"-like radiation pattern.
    All arrays are of shape (Nteta, Nphi)

    kwargs are passed to matplotlib's `pyplot.pcolormesh`.

    Parameters
    ----------
    tetalist : 2D-`numpy.ndarray`, float
        teta angles

    philist : 2D-`numpy.ndarray`, float
        phi angles

    Iff : 2D-`numpy.ndarray`, float
        Intensities at (teta, phi) positions

    degrees : bool, default: True
        Transform polar angle to degrees for plotting (False: radians)

    ax : matplotlib `axes`, default: None (=create new)
        optinal axes object (mpl) to plot into

    show : bool, default: True
        whether to call `pyplot.show()`


    Returns
    -------
        result of matplotlib's `pyplot.pcolormesh`
    """
    if ax is None:
        ax = plt.gca()
        if ax.name == 'polar':
            pass
        else:
            ax = plt.subplot(polar=True)

    Nteta, Nphi = I.shape

    ## --- for plotting only: Add 360degrees (copy of 0 degrees)
    teta = np.concatenate([theta.T, [theta.T[-1]]]).T
    phi = np.concatenate([phi.T, [np.ones(phi.T[-1].shape) * 2*np.pi]]).T - np.pi/float(Nphi)
    I = np.concatenate([I.T, [I.T[-1]]]).T

    ## --- other parameters
    if degrees:
        conv_factor = 180./np.pi
    else:
        conv_factor = 1

    if 'edgecolors' not in kwargs:
        kwargs['edgecolors']='face'

    ## --- plot
    im = ax.pcolormesh(phi, teta*conv_factor, I, **kwargs)

    if show:
        plt.colorbar(im, ax=ax)
        plt.show()

    return im







##----------------------------------------------------------------------
##               Oszillating Field Animation
##----------------------------------------------------------------------
def animate_vectorfield(NF, Nframes=50, projection='auto', doQuiver=True, doColor=False,
                     alphaColor=1.0, scale=5., clim=None,
                     kwargs={'cmin':0.3}, kwargsColor={'fieldComp':'Ex'}, ax=None,
                     t_start=0, frame_list=None, show=True):
    """Create animation of oszillating complex nearfield (2D-Projection)

    Parameters
    ----------
    NF : list of 6-tuples
        Nearfield definition. `np.array`, containing 6-tuples:
        (X,Y,Z, Ex,Ey,Ez), the field components being complex (use e.g.
        :func:`.tools.get_field_as_list`).

    Nframes : int, default: 50
          number of frames per oscillation cycle

    projection : str, default: 'auto'
        which 2D projection to plot: "auto", "XY", "YZ", "XZ"

    doQuiver : bool, default: True
          plot field as quiverplot.

    doColor : bool, default: True
          plot field as colorplot (real part)

    alphaColor : float, default: 0.5
          alpha value of colormap

    kwargs : dict, default: {}
          are passed to :func:`.vectorfield` (if doQuiver)

    kwargsColor : dict, default: {}
          are passed to :func:`.vectorfield_color` (if doColor)

    ax : optional matplotlib `axes`, default: None
          optional axes object (mpl) to plot into

    t_start : int, default: 0
        time-step to start animation at (=frame number)

    frame_list : list, default: None
        optional list of frame indices to use for animation. Can be used to
        animate only a part of the time-harmonic cycle.

    show : bool, default: True
          directly show animation

    Returns
    -------
        im_ani : Animation as matplotlib "ArtistAnimation" object


    Notes
    -----
        You can save the animation as video-file using:
        im_ani.save('video.mp4', writer="ffmpeg", codec='h264', bitrate=1500).
        See also matplotlib's documentation of the `animation` module for more info.

    """

    NF = NF.T
    if len(NF) != 6:
        raise ValueError("wrong shape of Nearfield Array. Must consist of 6-tuples: [x,y,z, Ex,Ey,Ez]")

    x,y,z = NF[0:3]

    if projection.lower() == 'auto':
        projection = _automatic_projection_select(x,y,z)

    ## get phase and length of complex field
    Exi = NF[3]
    Exr = np.absolute(Exi)
    Ax  = np.angle(Exi)

    Eyi = NF[4]
    Eyr = np.absolute(Eyi)
    Ay  = np.angle(Eyi)

    Ezi = NF[5]
    Ezr = np.absolute(Ezi)
    Az  = np.angle(Ezi)

    scaleF = float((Exr.max()+Eyr.max()+Ezr.max()))
    Exr /= scaleF
    Eyr /= scaleF
    Ezr /= scaleF

    ## create list of timesteps
    import matplotlib.pyplot as plt
    alambda = 100
    omega = 2*np.pi/alambda

    if show:
        fig = plt.figure()
    else: fig = plt.gcf()

    ax = ax or plt.subplot(aspect="equal")

    framnumbers = np.linspace(t_start, alambda+t_start, Nframes)
    if frame_list is not None:
        framnumbers = framnumbers[frame_list]

    ims = []
    for t in framnumbers:
        plotlist = []
        NF = np.transpose([x,y,z, (Exr * np.exp(1j*(Ax - omega*t))),
                                  (Eyr * np.exp(1j*(Ay - omega*t))),
                                  (Ezr * np.exp(1j*(Az - omega*t)))])
        if doColor:
            pt2 = vectorfield_color(NF, alpha=alphaColor, projection=projection,
                                    cmap='seismic', show=False, clim=clim, **kwargsColor)
            plotlist.append(pt2)
        if doQuiver:
            pt1 = vectorfield(NF, projection=projection, scale=scale,
                              overrideMaxL=True, show=False, **kwargs)
            plotlist.append(pt1)

        ims.append( tuple(plotlist) )


    import matplotlib.animation as animation
    im_ani = animation.ArtistAnimation(fig, ims, interval=50, repeat_delay=0,
                                       blit=True, repeat=True)

    if show:
        plt.show()

    return im_ani









