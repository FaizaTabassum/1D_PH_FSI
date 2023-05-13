import sv
import vtk
import random, os


def create_solid_from_path_modified(src_path, initial_radius):
    """
    Creates a lofted solid from the provided source path with circular contours
    with radii +/- 0.25 from initial_radius.

    Args:
     src_path (pathplanning.Path): Source path.
     initial_radius (double): Initial "average" radius to use.
    Returns:
     tuple(list[sv.segmentation.Circle], sv.modeling.Model): Lofted model and contours.

    Source: copied from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py
    """

    # Store the path position points.
    path_pos_points = src_path.get_curve_points()

    # Create contours from the points.
    contours = []                # List of contour objects created.
    contour_pds = []             # List of polydata objects created from the contours.
    # Extract every 10'th path point and create a circular contour around it.
    i=0
    for id in range(int(len(path_pos_points) / 10)):
        path_point_id = id * 10

        # Randomize the radius and create the circular contour. Coords for the
        # center must be defined in absolute 3D space, so we must grab the real
        # position point from the path data.
        radius = initial_radius[i]
        i+=10

        # Create a new circular contour object.
        contour = sv.segmentation.Circle(radius = radius,
                               center = path_pos_points[path_point_id],
                               normal = src_path.get_curve_tangent(path_point_id))


        # Extract a polydata object from the created contour and save it in the list.
        contours.append(contour)
        contour_pds.append(contour.get_polydata())

    # Resample and align the contour polydata objects to ensure that all
    # contours contain the same quantity of points and are all rotated such that
    # the ids of each point in the contours are in the same position along the
    # contours for lofting.
    num_samples = 25    # Number of samples to take around circumference of contour.
    use_distance = True # Specify option for contour alignment.
    for index in range(0, len(contour_pds)):
        # Resample the current contour.
        contour_pds[index] = sv.geometry.interpolate_closed_curve(
                                            polydata=contour_pds[index],
                                            number_of_points=num_samples)

        # Align the current contour with the previous one, beginning with the
        # second contour.
        if not index is 0:
            contour_pds[index] = sv.geometry.align_profile(
                                                contour_pds[index - 1],
                                                contour_pds[index],
                                                use_distance)

    # Loft the contours.
    # Set loft options.
    options = sv.geometry.LoftOptions()
    # Use linear interpolation between spline sample points.
    options.interpolate_spline_points = True
    # Set the number of points to sample a spline if
    # using linear interpolation between sample points.
    options.num_spline_points = 50
    # The number of longitudinal points used to sample splines.
    options.num_long_points = 200

    # Loft solid.
    lofted_surface = sv.geometry.loft(polydata_list=contour_pds, loft_options=options)

    # Create a new solid from the lofted solid.
    lofted_model = sv.modeling.PolyData()
    lofted_model.set_surface(surface=lofted_surface)

    # Cap the lofted volume.
    capped_model_pd = sv.vmtk.cap(surface=lofted_model.get_polydata(),
                                  use_center=False)
    # capped_model_pd = sv.vmtk.cap_with_ids(surface=lofted_model.get_polydata(),
    #                                     fill_id=1, increment_id=True)
    # path_lofted_capped_name = path_lofted_name + "_capped"
    # VMTKUtils.Cap_with_ids(path_lofted_name, path_lofted_capped_name, 0, 0)
    # solid.SetVtkPolyData(path_lofted_capped_name)
    # num_triangles_on_cap = 150
    # solid.GetBoundaryFaces(num_triangles_on_cap)

    # Import the capped model PolyData into model objects.
    capped_model = sv.modeling.PolyData()
    capped_model.set_surface(surface=capped_model_pd)

    return (contours, capped_model)

def create_solid_from_path(src_path, initial_radius):
    """
    Creates a lofted solid from the provided source path with circular contours
    with radii +/- 0.25 from initial_radius.

    Args:
     src_path (pathplanning.Path): Source path.
     initial_radius (double): Initial "average" radius to use.
    Returns:
     tuple(list[sv.segmentation.Circle], sv.modeling.Model): Lofted model and contours.

    Source: copied from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py
    """

    # Store the path position points.
    path_pos_points = src_path.get_curve_points()

    # Create contours from the points.
    prev_radius = initial_radius # Last radius from which to add/subtract a random number.
    contours = []                # List of contour objects created.
    contour_pds = []             # List of polydata objects created from the contours.
    # Extract every 10'th path point and create a circular contour around it.
    for id in range(int(len(path_pos_points) / 10)):
        path_point_id = id * 10

        # Randomize the radius and create the circular contour. Coords for the
        # center must be defined in absolute 3D space, so we must grab the real
        # position point from the path data.
        radius = prev_radius + 0.5 * (random.random() - 0.5)
        prev_radius = radius

        # Create a new circular contour object.
        contour = sv.segmentation.Circle(radius = radius,
                               center = path_pos_points[path_point_id],
                               normal = src_path.get_curve_tangent(path_point_id))


        # Extract a polydata object from the created contour and save it in the list.
        contours.append(contour)
        contour_pds.append(contour.get_polydata())

    # Resample and align the contour polydata objects to ensure that all
    # contours contain the same quantity of points and are all rotated such that
    # the ids of each point in the contours are in the same position along the
    # contours for lofting.
    num_samples = 25    # Number of samples to take around circumference of contour.
    use_distance = True # Specify option for contour alignment.
    for index in range(0, len(contour_pds)):
        # Resample the current contour.
        contour_pds[index] = sv.geometry.interpolate_closed_curve(
                                            polydata=contour_pds[index],
                                            number_of_points=num_samples)

        # Align the current contour with the previous one, beginning with the
        # second contour.
        if not index is 0:
            contour_pds[index] = sv.geometry.align_profile(
                                                contour_pds[index - 1],
                                                contour_pds[index],
                                                use_distance)

    # Loft the contours.
    # Set loft options.
    options = sv.geometry.LoftOptions()
    # Use linear interpolation between spline sample points.
    options.interpolate_spline_points = True
    # Set the number of points to sample a spline if
    # using linear interpolation between sample points.
    options.num_spline_points = 50
    # The number of longitudinal points used to sample splines.
    options.num_long_points = 200

    # Loft solid.
    lofted_surface = sv.geometry.loft(polydata_list=contour_pds, loft_options=options)

    # Create a new solid from the lofted solid.
    lofted_model = sv.modeling.PolyData()
    lofted_model.set_surface(surface=lofted_surface)

    # Cap the lofted volume.
    capped_model_pd = sv.vmtk.cap(surface=lofted_model.get_polydata(),
                                  use_center=False)
    # capped_model_pd = sv.vmtk.cap_with_ids(surface=lofted_model.get_polydata(),
    #                                     fill_id=1, increment_id=True)
    # path_lofted_capped_name = path_lofted_name + "_capped"
    # VMTKUtils.Cap_with_ids(path_lofted_name, path_lofted_capped_name, 0, 0)
    # solid.SetVtkPolyData(path_lofted_capped_name)
    # num_triangles_on_cap = 150
    # solid.GetBoundaryFaces(num_triangles_on_cap)

    # Import the capped model PolyData into model objects.
    capped_model = sv.modeling.PolyData()
    capped_model.set_surface(surface=capped_model_pd)

    return (contours, capped_model)

def create_path_from_points_list(path_points_list):
    """
    Inputs:
        list or np.array path_points_list
            = [list of coordinates of path points]
                where coordinates is a list of the form, [x, y, z]
    Returns:
        sv.pathplanning.Path() path
    """
    path = sv.pathplanning.Path()
    for point in path_points_list:
        path.add_control_point(point)
    return path

def create_segmentations_from_path_and_radii_list(path, radii_list):
    """
    Inputs:
        sv.pathplanning.Path() path
        list or np.array path_points_list
            = [list of coordinates of path points]
                where coordinates is a list of the form, [x, y, z]
        list or np.array radii_list
            = [radius for each point in path_points_list]
    Returns:
        list segmentations
            = [list of sv.segmentation.Circle contour objects]
    """
    path_points_list = path.get_control_points()
    path_curve_points = path.get_curve_points() # path curve points are not the same as the path control points (used in function create_path_from_points_list). There are more path curve points than path control points. The path curve points are the points that comprise the path spline
    segmentations = []
    for path_point_id in range(len(path_points_list)):
        contour = sv.segmentation.Circle(radius = radii_list[path_point_id], center = path_points_list[path_point_id], normal = path.get_curve_tangent(path_curve_points.index(path_points_list[path_point_id])))
        segmentations.append(contour)
    return segmentations
