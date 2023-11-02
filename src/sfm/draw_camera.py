import numpy as np

from sfm.arrow_3d import Arrow3D

def drawCamera(ax, position, direction, length_scale = 1, head_size = 10, 
        equal_axis = True, set_ax_limits = True):
    # Draws a camera consisting of arrows into a 3d Plot
    # ax            axes object, creates as follows
    #                   fig = plt.figure()
    #                   ax = fig.add_subplot(projection='3d')
    # position      np.array(3,) containing the camera position
    # direction     np.array(3,3) where each column corresponds to the [x, y, z]
    #               axis direction
    # length_scale  length scale: the arrows are drawn with length
    #               length_scale * direction
    # head_size     controls the size of the head of the arrows
    # equal_axis    boolean, if set to True (default) the axis are set to an 
    #               equal aspect ratio
    # set_ax_limits if set to false, the plot box is not touched by the function

    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='r')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 0]],
                [position[1], position[1] + length_scale * direction[1, 0]],
                [position[2], position[2] + length_scale * direction[2, 0]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='g')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 1]],
                [position[1], position[1] + length_scale * direction[1, 1]],
                [position[2], position[2] + length_scale * direction[2, 1]],
                **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(mutation_scale=head_size, arrowstyle='-|>', color='b')
    a = Arrow3D([position[0], position[0] + length_scale * direction[0, 2]],
                [position[1], position[1] + length_scale * direction[1, 2]],
                [position[2], position[2] + length_scale * direction[2, 2]],
                **arrow_prop_dict)
    ax.add_artist(a)

    if not set_ax_limits:
        return

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_xlim([min(xlim[0], position[0]), max(xlim[1], position[0])])
    ax.set_ylim([min(ylim[0], position[1]), max(ylim[1], position[1])])
    ax.set_zlim([min(zlim[0], position[2]), max(zlim[1], position[2])])
    ax.legend(['points','x', 'y', 'z'])
    
    # This sets the aspect ratio to 'equal'
    if equal_axis:
        ax.set_box_aspect((np.ptp(ax.get_xlim()),
                       np.ptp(ax.get_ylim()),
                       np.ptp(ax.get_zlim())))
