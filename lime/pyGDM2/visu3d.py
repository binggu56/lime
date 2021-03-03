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
Collection of 3D visualization tools for pyGDM2

all 3D plotting is done using `mayavi`

"""
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from . import tools

## Mayavi Imports
from mayavi import mlab
from mayavi.sources.builtin_surface import BuiltinSurface
from mayavi.modules.surface import Surface
from mayavi.filters.transform_data import TransformData
#from tvtk.util import ctf
#from tvtk.tools import visual








#==============================================================================
# predefined mayavi plotting functions
#==============================================================================
#==============================================================================
# Substrate
#==============================================================================
def _draw_rect(X0,X1, Y0,Y1, Z0,Z1, rotate_axis=(1,0,0), rotate_angle=(0),
                opacity=1, color=(.5,.5,.5)):
    engine = mlab.get_engine()
    def rotMat3D(axis, angle, tol=1e-12):
        """Return the rotation matrix for 3D rotation by angle `angle` degrees about an
        arbitrary axis `axis`.
        """
        t = np.radians(angle)
        x, y, z = axis
        R = (np.cos(t))*np.eye(3) +\
            (1-np.cos(t))*np.matrix(((x**2,x*y,x*z),(x*y,y**2,y*z),(z*x,z*y,z**2))) + \
            np.sin(t)*np.matrix(((0,-z,y),(z,0,-x),(-y,x,0)))
        R[np.abs(R)<tol]=0.0
        return R

    # Main code
    # Add a cubic builtin source
    rect_src = BuiltinSurface()
    engine.add_source(rect_src)
    rect_src.source = 'cube'
    rect_src.data_source.center = np.array([ (X1+X0)/2.,  (Y1+Y0)/2.,  (Z1+Z0)/2.])
    rect_src.data_source.x_length=X1-X0
    rect_src.data_source.y_length=Y1-Y0
    rect_src.data_source.z_length=Z1-Z0
    #~ rect_src.data_source.capping = False
    #~ rect_src.data_source.resolution = 250


    # Add transformation filter to rotate rect about an axis
    transform_data_filter = TransformData()
    engine.add_filter(transform_data_filter, rect_src)
    Rt = np.eye(4)
    Rt[0:3,0:3] = rotMat3D(rotate_axis, rotate_angle) # in homogeneous coordinates
    Rtl = list(Rt.flatten()) # transform the rotation matrix into a list
    transform_data_filter.transform.matrix.__setstate__({'elements': Rtl})
    transform_data_filter.widget.set_transform(transform_data_filter.transform)
    transform_data_filter.filter.update()
    transform_data_filter.widget.enabled = False   # disable the rotation control further.

    # Add surface module to the rect source
    rect_surface = Surface()
    engine.add_filter(rect_surface, transform_data_filter)
    # add color property
    rect_surface.actor.property.color = color
    rect_surface.actor.property.opacity = opacity
    
    return rect_surface








########################################################################
##                      VISUALIZATION FUNCTIONS
########################################################################
def structure(struct, scale=0.75, abs_scale=False, 
              tit='', color='auto', mode='cube', 
              draw_substrate=True, substrate_size=2.0, 
              substrate_color=(0.8, 0.8, 0.9), substrate_opacity=0.5,
              axis_labels=True, material_labels=False,
              show=True, **kwargs):
    """plot structure in 3d
    
    plot the structure "struct" using 3d points. Either from list of 
    coordinates, or using a simulation definition dict as input.
    
    Parameters
    ----------
      - struct:    either simulation-dictionary or list of 3d coordinate tuples
      - scale:     symbol scaling in units of stepsize (default 0.75)
      - abs_scale: enable absolute scaling, override internal scale calculation (default: False)
      - color:     Color of scatterplot. Either "auto", or mayavi2-compatible color.
      - mode:      3d symbols for plotting meshpoints. see `mlab.points3d`. e.g. 'cube' or 'sphere' (default 'cube')
      - draw_substrate: Whether or not to draw a substrate (default: True)
      - substrate_size: size of substrate with respect to structure extensions (default: 2.0)
      - substrate_color: default (0.8, 0.8, 0.9)
      - substrate_opacity: default 0.5
      - axis_labels: whether to show the X/Y/Z dimensions of the nano-object (default: True)
      - material_labels: whether or not to add material names labels (default: False)
      - show:      directly show plot (default True)
      - kwargs:    are passed to `mlab.points3d`
    
    """
    from pyGDM2 import structures
    X,Y,Z = tools.get_geometry(struct)
    step = tools.get_step_from_geometry(struct)
    
    ## -- set scatter-scaling depending on structure size
    maxdist = max([max(X)-min(X), max(Y)-min(Y)])
    if not abs_scale: scale = step*scale
    
    ## -- colors for subsets with different materials
    if color == 'auto':
        if hasattr(struct, "struct") or type(struct) == structures.struct:
            if hasattr(struct, "struct"):
                struct = struct.struct
        
            if hasattr(struct.material, '__iter__'):
                materials = [s.__name__ for s in struct.material]
                if len(set(materials)) > 1:
                    material = np.array(materials)
                    different_materials = np.unique(materials)
                    indices_subsets = []
                    for struct_fraction in different_materials:
                        indices_subsets.append(np.arange(len(material))[material==struct_fraction])
                else:
                    color = (0.3,0.3,0.3)
            else:
                color = (0.3,0.3,0.3)
        else:
            color = (0.3,0.3,0.3)
    
    
    
    
    # 3D-Structure
    if show: 
        mlab.figure( bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.,0.,0.) )
    
    if draw_substrate:
        X0 = min(X) - (substrate_size/2. - 0.5) * maxdist
        Y0 = min(Y) - (substrate_size/2. - 0.5) * maxdist
        X1 = max(X) + (substrate_size/2. - 0.5) * maxdist
        Y1 = max(Y) + (substrate_size/2. - 0.5) * maxdist
        Z0 = - step
        Z1 = - step/10.
        _draw_rect(X0,X1, Y0,Y1, Z0,Z1, opacity=substrate_opacity, color=substrate_color)
    
    ## draw structure
    ## --- if multi-material: set maker colors
    if color != 'auto':
        im = mlab.points3d(X,Y,Z, mode=mode, scale_factor=scale, color=color, **kwargs)
    else:
        import matplotlib.pyplot as plt
        colors = [plt.cm.colors.to_rgb('C{}'.format(i)) for i in range(1,10)] * int(2 + len(indices_subsets)/9.)
        im = []
        for i, idx in enumerate(indices_subsets):
            col = colors[i]
            im.append(mlab.points3d(X[idx],Y[idx],Z[idx], mode=mode, scale_factor=scale, color=col, **kwargs))
            if material_labels:
                ## add space for approximately same text length and size
                N_max_char = max([len(l) for l in different_materials])
                mat_label = different_materials[i] + " "*2*(N_max_char-len(different_materials[i]))
                mlab.text(0.02, 0.95-0.05*i, mat_label,
                                color=col, opacity=0.8)
    
    if axis_labels: 
        mlab.axes(xlabel='X (nm)', ylabel='Y (nm)', zlabel='Z (nm)')
    if show: 
        mlab.title(tit)
        mlab.show()
    
    return im






    
def vectorfield(NF, struct=None, scale=1.5, abs_scale=False, 
                tit='', complex_part='real', clim=[0.0, 1.0],
                axis_labels=True, show=True, **kwargs):
    """3d quiverplot of nearfield
    
    Parameters
    ----------
     - NF:       Nearfield definition. `np.array`, containing 6-tuples:
                   (X,Y,Z, Ex,Ey,Ez), the field components being complex.
     - struct:   optional structure definition (if field is supplied in 3-tuple 
                 form without coordinates). Either `simulation` object, or list
                 of coordinate (x,y,z) tuples 
     - scale:     symbol scaling in units of stepsize (default 0.75)
     - abs_scale: enable absolute scaling, override internal scale calculation (default: False)
     - complex_part: Which part of complex field to plot. 
                     Either 'real' or 'imag'. (default: 'real')
     - axis_labels: whether to show the X/Y/Z dimensions of the nano-object (default: True)
     - show:     whether to directly show the figure (default: True)
    
    All other keyword arguments are passed to mlab's `quiver3d`.
    """
    ## case: provided structure+NF separately: NF may contain field_config dict as first element
    if len(NF) == 2:
        NF = NF[1]
    
    if len(NF.T) == 6:
        X,Y,Z, UXcplx,UYcplx,UZcplx = np.transpose(NF).real
    elif len(NF.T) == 3 and struct is not None:
        UXcplx,UYcplx,UZcplx = np.transpose(NF)
        X,Y,Z = tools.get_geometry(struct).real
    else:
        raise ValueError("Error: Wrong number of columns in vector field. Expected (Ex,Ey,Ez)-tuples + `simulation` object or (x,y,z, Ex,Ey,Ez)-tuples.")
    
    if complex_part.lower() == "real":
        Ex,Ey,Ez = UXcplx.real, UYcplx.real, UZcplx.real
    elif complex_part.lower() == "imag":
        Ex,Ey,Ez = UXcplx.imag, UYcplx.imag, UZcplx.imag
    else:
        raise ValueError("Error: Unknown `complex_part` argument value. Must be either 'real' or 'imag'.")
    S = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    
    step = tools.get_step_from_geometry(np.array([X,Y,Z]).T)
    if not abs_scale:
        scale = step*scale
    else:
        scale = scale
    
    ## 3D-plot
    if show: 
        mlab.figure( bgcolor=(0.2,0.2,0.2), fgcolor=(1.0, 1.0, 1.0) )
    
    s = mlab.quiver3d(X,Y,Z, Ex,Ey,Ez, scalars=S, scale_factor=scale, 
                      vmin=clim[0], vmax=clim[1], **kwargs)
    
    if axis_labels: 
        mlab.axes(xlabel='X (nm)', ylabel='Y (nm)', zlabel='Z (nm)')
    if show: 
        mlab.title(tit)
        mlab.show()
    
    return s


def vectorfield_by_fieldindex(sim, field_index, **kwargs):
    """Wrapper to :func:`vectorfield`, using simulation object and fieldindex as input
    
    Parameters
    ----------
    sim : `simulation`
        instance of :class:`.core.simulation`
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
        
    All other keyword arguments are passed to :func:`vectorfield`.
    """
    NF = sim.E[field_index]
    im = vectorfield(NF, sim, **kwargs)
    return im
    






## 3D colorplot
def vectorfield_color(NF, complex_part='real', 
                      fieldComp='I', scale=0.5, abs_scale=False, mode='sphere',
                      tit='', axis_labels=True, show=True, **kwargs):
    """plot of scalar electric field data as 3D colorplot
    
    `vectorfield_color` is using `mlab.quiver3d` to plot colored data-points in
    order to be able to fix the size of the points while varying the color-code
    
    Parameters
    ----------
     - NF:       Nearfield definition. `np.array`, containing 6-tuples:
                   (X,Y,Z, Ex,Ey,Ez), the field components being complex.
     - complex_part: Complex part to plot. Either 'real' or 'imag' 
                       (default 'real')
     - fieldComp: default='I'. Which component to use. One of ["I", "Ex", "Ey", "Ez"].
                    if "I" is used, `complex_part` argument has no effect.
     - scale:     symbol scaling in units of stepsize (default 0.75)
     - abs_scale: enable absolute scaling, override internal scale calculation (default: False)
     - mode:     which glyph to use (default: 'sphere', might also be e.g. 'cube')
     - tit:      optional plot title (default '')
     - axis_labels: whether to show the X/Y/Z dimensions of the nano-object (default: True)
     - show:     whether to directly show the figure (default: True)
     
    other kwargs are passed to mlabs's `quiver3d`
    """
    if len(NF.T) == 6:
        X,Y,Z, Ex,Ey,Ez = NF.T
        X,Y,Z = X.real, Y.real, Z.real
    else:
        raise ValueError("Error: Field list must contain tuples of exactly 6 elements.")
    
    if fieldComp.lower() != 'i':
        if complex_part.lower() == "real":
            Ex, Ey, Ez = Ex.real, Ey.real, Ez.real
        elif complex_part.lower() == "imag":
            Ex, Ey, Ez = Ex.imag, Ey.imag, Ez.imag
        else:
            raise ValueError("Error: Unknown `complex_part` argument value. Must be either 'real' or 'imag'.")
    
    
    if fieldComp.lower() == 'i':
        EF = np.abs(Ex**2 + Ey**2 + Ez**2)
    elif fieldComp.lower() == 'ex':
        EF = Ex
    elif fieldComp.lower() == 'ey':
        EF = Ey
    elif fieldComp.lower() == 'ez':
        EF = Ez
    
    step = tools.get_step_from_geometry(np.array([X,Y,Z]).T)
    if not abs_scale:
        scale = step*scale
    else:
        scale = scale
    
    ## 3D-plot
    if show: 
        mlab.figure( bgcolor=(0.2,0.2,0.2), fgcolor=(1.0, 1.0, 1.0) )
    
    s = np.ones(X.shape)
    o = np.zeros(X.shape)
    pts = mlab.quiver3d(X,Y,Z, s, o, o, scalars=EF, mode=mode, scale_factor=scale, **kwargs) 
    pts.glyph.color_mode = 'color_by_scalar'
    pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    
    
    if axis_labels: 
        mlab.axes(xlabel='X (nm)', ylabel='Y (nm)', zlabel='Z (nm)')
    if show: 
        mlab.title(tit)
        mlab.show()
    
    return pts






def vectorfield_color_by_fieldindex(sim, field_index, **kwargs):
    """Wrapper to :func:`vectorfield_color`, using simulation object and fieldindex as input
    
    Parameters
    ----------
    sim : `simulation`
        instance of :class:`.core.simulation`
    
    field_index : int
        index of evaluated self-consistent field to use for calculation. Can be
        obtained for specific parameter-set using :func:`.tools.get_closest_field_index`
        
    All other keyword arguments are passed to :func:`vectorfield_color`.
    """
    x, y, z = sim.struct.geometry.T
    Ex, Ey, Ez = sim.E[field_index][1].T
    NF = np.transpose([x,y,z, Ex,Ey,Ez])
    
    im = vectorfield_color(NF, **kwargs)
    return im


def scalarfield(NF, **kwargs):
    """Wrapper to :func:`vectorfield_color`, using scalar data tuples (x,y,z,S) as input
    
    Parameters
    ----------
    NF : list of 4-tuples
        list of tuples (x,y,z,S).
        Alternatively, the scalar-field can be passed as list of 2 lists 
        containing the x/y positions and scalar values, respectively.
        ([xy-values, S], with xy-values: list of 2-tuples [x,y]; S: list of 
        scalars). This format is returned e.g. by 
        :func:`.tools.calculate_rasterscan`.
        
    All other keyword arguments are passed to :func:`vectorfield_color`.
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







###----------------------------------------------------------------------
###               FARFIELD (INTENSITY)
###----------------------------------------------------------------------
#def _sphericToCarthesian(TETA, PHI, INTENS):
#    """Transfrom from spherical to carthesian coordinates
#    
#      input:
#        TETA:    List of teta angles (one entry for each intensity)
#        PHI:     List of phi angles (one entry for each intensity)
#        INTENS:  List of intensities
#    
#      output:
#        EFFX,EFFY,EFFZ:  X,Y,Z values in carthesian space
#        norm:            normalization factor for dataset
#    """
#    dat = np.transpose([TETA, PHI, INTENS])
#    EFFX=[]
#    EFFY=[]
#    EFFZ=[]
#    I=[]; I2=[]
#    Theta=[]; Theta2=[]
#    lastT=9999
#    for d in dat:
#        if d[0]!=lastT:
#            lastT=d[0]
#            EFFX.append([]); EFFY.append([]); EFFZ.append([])
#        teta      = d[0]
#        phi       = d[1]
#        Intensity = d[2]
#        
#        EFFX[-1].append(Intensity * np.sin(teta) * np.cos(phi))
#        EFFY[-1].append(Intensity * np.sin(teta) * np.sin(phi))
#        EFFZ[-1].append(Intensity * np.cos(teta))
#    EFFX=np.array(EFFX); EFFY=np.array(EFFY); EFFZ=np.array(EFFZ)
#    norm = max([np.max(EFFX),np.max(EFFY),np.max(EFFZ)])
#    return EFFX,EFFY,EFFZ,norm
#
#
#def plotFarfield3D(TETA, PHI, INTENS, TETAc=None, PHIc=None, INTENSc=None, norm=1, 
#                   title='', show=True):
#    """Plot 3D Farfield Radiation Pattern using mayavi
#    
#    Plot a radiation pattern in 3D space.
#    
#      input:
#        TETA:    List of teta angles (one entry for each intensity)
#        PHI:     List of phi angles (one entry for each intensity)
#        INTENS:  List of intensities
#        title:   (optional) title of plot
#        show:     directly show plot (default True)
#        
#      return:
#        None
#      
#    """
#    
#    if show: mlab.figure(bgcolor=(.9,.9,.9), size=(800,600))
#    
#    EX,EY,EZ,n = _sphericToCarthesian(TETA, PHI, INTENS)
#    norm = float(n)/float(norm)
#    if TETAc is not None:
#        EXwire,EYwire,EZwire,_n = _sphericToCarthesian(TETAc, PHIc, INTENSc)
#        mlab.mesh(EXwire/norm,EYwire/norm,EZwire/norm, color=(0,0,1), representation='wireframe', opacity=0.1, line_width=0.1)
#    
#    
#    mlab.mesh(EX/norm,EY/norm,EZ/norm, colormap="Blues", opacity=0.3)
#    
#    
#    if show: mlab.show()
#
#

##----------------------------------------------------------------------
##               Oszillating Field Animation
##----------------------------------------------------------------------
def animate_vectorfield(NF, Nframes=50, show=True, scale=1.5, abs_scale=False, 
                        draw_struct=False,
                        draw_substrate=True, substrate_size=2.0, 
                        clim=[0.0, 1.0],
                        figsize=(600, 400), save_anim=False, 
                        fig=None, view=[45, 45],
                        rotate_azimuth=0, rotate_elevation=0,
                        ffmpeg_args="", mov_file="pygdm_animated_field.mp4",
                        tmp_file_prefix='_img_pygdmanim_tmp',
                        save_cycles=1, t_start=0, frame_list=None,
                        **kwargs):
    """animate vector-field in 3d
    
    animate the time-harmonic vectors of an electromagnetic field
    
    Parameters
    ----------
    NF : list of 4-tuples
        list of tuples (x,y,z,S).
        Alternatively, the scalar-field can be passed as list of 2 lists 
        containing the x/y positions and scalar values, respectively.
        ([xy-values, S], with xy-values: list of 2-tuples [x,y]; S: list of 
        scalars). This format is returned e.g. by 
        :func:`.tools.calculate_rasterscan`.
        
    Nframes : 
    
    
    scale : float, default: 1.5
        vector scaling in units of stepsize
        
    abs_scale : bool, default: False
        enable absolute scaling, override internal scale calculation
    
    draw_struct : bool, default: False
        whether to draw the structure, assuming that every point in 'NF' is a geometry meshpoint
        
    draw_substrate : bool, default: True
        whether to draw a sketch of a substrate
        
    substrate_size : float, default: 2.0
        size of substrate, if drawn. Factor relative to structure extension
        
    clim : 2 element list, default: [0.0, 1.0]
        range to use from colormap
    
    figsize : 2-tuple, default: (600, 400)
        plot size in pixels (X,Y)
    
    save_anim : bool, default: False
        whether to save a video-file of animation (requires the program 'ffmpeg')
    
    fig : (optional) mlab figure, default: None 
        optional mayavi2 "mlab" figure, which can already contain other plotted elements
    
    view : tuple, default (45, 45)
        view perspective, passed to :func:`mlab.view`: (azimuth, elevation, distance, [r_x,r_y,r_z])
    
    rotate_azimuth, rotate_elevation : int, int, default: 0, 0
        optional total rotation angle of camera during animation. Camera
        will rotate from initial angle alpha_0 to alpha_0 + rotate_angle.
        Should be used together with `save_cycles`>1.
    
    ffmpeg_args : str, default: "-b:v 1.5M -c:v libx264"
        string of command line arguments passed to 'ffmpeg' (if save_anim == True)
    
    mov_file : str, default: "pygdm_animated_field.mp4"
        movie file to save animation (if save_anim == True)
    
    tmp_file_prefix : str, default: '_img_pygdmanim_tmp'
        tmp-file prefix for movie-saving. Will be deleted after encoding of the movie.
    
    save_cycles : int, default: 1
        number of harmonic field cycles to do for animation until restart. 
        Useful if camera is slowly rotated during animation.
    
    t_start : int, default: 0
        time-step to start animation at (=frame number)
        
    frame_list : list, default: None
        optional list of frame indices to use for animation. Can be used to 
        animate only a part of the time-harmonic cycle.
        
    **kwargs : 
        other keyword arguments are passed to :func:`mlab.quiver3d`
    
    
    Returns
    -------
    
    None
    
    """
    
    NF = NF.T
    if len(NF) != 6:
        raise ValueError("wrong shape of Nearfield Array. Must consist of 6-tuples: [x,y,z, Ex,Ey,Ez]")
    
    x,y,z = NF[0:3].real
    maxdist = max([max(x)-min(x), max(y)-min(y)])
    
    step = tools.get_step_from_geometry(NF[0:3].real.T)
    if not abs_scale:
        scale_quiver = step*scale
    else:
        scale_quiver = scale
        
    
    ## --- phase and length of complex field
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
    
    ## --- create list of timesteps
    alambda = 100.
    omega = 2*np.pi/float(alambda)
    
    framnumbers = np.linspace(t_start, alambda+t_start, Nframes)
    if frame_list is not None:
        framnumbers = framnumbers[frame_list]
    
    ims = []
    Emax = []
    for t in framnumbers:    
        Ex = (Exr * np.cos((Ax - omega*t))).real
        Ey = (Eyr * np.cos((Ay - omega*t))).real
        Ez = (Ezr * np.cos((Az - omega*t))).real
        E = np.sqrt( Ex**2 +Ey**2 + Ez**2)            
        Emax.append(E.max())
        ims.append( [x,y,z, Ex,Ey,Ez, E] )
    for i, _tmp in enumerate(Emax):
        ims[i][-1] /= max(Emax)
    

#==============================================================================
#     The actual plot
#==============================================================================
    fig_new = False
    if fig is None:
        fig_new = True
        fig = mlab.figure( size=figsize, bgcolor=(1.0, 1.0, 1.0), fgcolor=(0.,0.,0.) )
    
    if draw_substrate:
        X0 = Y0 = min([min(x), min(y)]) - (substrate_size/2. - 0.5) * maxdist
        X1 = Y1 = max([max(x), max(y)]) + (substrate_size/2. - 0.5) * maxdist
        Z0 = - step
        Z1 = - step/10.
        _draw_rect(X0,X1, Y0,Y1, Z0,Z1, opacity=0.2, color=(0.3, 0.3, 0.9))
    
    if draw_struct:
        mlab.points3d(x,y,z, mode='cube', scale_factor=step*scale*0.25, 
                      color=(0.1,0.1,0.1), opacity=0.15)
    
    ## the field quiver-plot
    D = ims[-1]
    p3d = mlab.quiver3d(D[0],D[1],D[2], D[3],D[4],D[5], scalars=D[6], 
                        scale_factor=scale_quiver, vmin=clim[0], vmax=clim[1], **kwargs)
    p3d.glyph.color_mode = 'color_by_scalar'
    
    mlab.view(*view)
    
    
#==============================================================================
#     The actual animation, plus ffmpeg mp4-save stuff
#==============================================================================
    azimuth_angle = view[0]
    elevation_angle = view[1]
    
    @mlab.animate(delay=40)
    def anim():
        j_rep = 0
        while True:
            for i, D in enumerate(ims):
                ## update data
                p3d.mlab_source.set(x=D[0],y=D[1],z=D[2], u=D[3],v=D[4],w=D[5])
                p3d.mlab_source.scalars = D[6]
                
                ## update view (if accordingly configured)
                if rotate_azimuth!=0 or rotate_elevation!=0:
                    view[0] = azimuth_angle + (i + len(ims)*j_rep)*rotate_azimuth
                    view[1] = elevation_angle + (i + len(ims)*j_rep)*rotate_elevation
                    mlab.view(*view)
                
                if save_anim:
                    mlab.savefig("{:0>6}{}.png".format(i + len(ims)*j_rep, tmp_file_prefix))
                yield
            j_rep +=1
            
            if save_anim and j_rep >= save_cycles:
                break
            
        ## --- assemble movie
        if save_anim:
            from subprocess import check_output
#            print(check_output(["cd {}".format(image_tmp_dir)], shell=True))
            print(check_output(["ffmpeg -f image2 -r 20 -i '%06d{}.png' {} {}".format(tmp_file_prefix, ffmpeg_args, mov_file)], shell=True))
            print(check_output(["rm *{}.png".format(tmp_file_prefix)], shell=True))
    
    anim()
    
    if show:
        mlab.show()









