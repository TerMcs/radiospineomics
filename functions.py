import cv2
import itertools
import math
import nrrd
import numpy as np
import scipy.stats as stats
import shapely

from shapely.ops import linemerge, unary_union, polygonize
from sklearn import mixture


def get_delta_si(masks, image):

    mean = np.mean(image)
    std = np.std(image)
    outliers_removed = image[image < mean + 3 * std]
    outliers_removed = outliers_removed[outliers_removed > mean - 3 * std]
    mean_image_intensity = np.mean(outliers_removed)

    int_mask_pixel_values = image[masks[1].astype(bool)]
    
    f = np.ravel(int_mask_pixel_values).astype(float)
    f=f.reshape(-1,1)
    g = mixture.GaussianMixture(n_components=2, max_iter=1000, random_state=10, covariance_type = 'full')
    g.fit(f)
    weights = g.weights_
    means = g.means_
    covars = g.covariances_

    x_axis = f.copy().ravel()
    x_axis.sort()

    y_axis0 = stats.norm.pdf(x_axis, float(means[0][0]), np.sqrt(float(covars[0][0][0])))*weights[0] # 1st gaussian
    y_axis1 = stats.norm.pdf(x_axis, float(means[1][0]), np.sqrt(float(covars[1][0][0])))*weights[1] # 2nd gaussian
    
    delta_si = np.abs(x_axis[np.argmax(y_axis0)] - x_axis[np.argmax(y_axis1)])
    normalised_delta_si = (delta_si / mean_image_intensity)

    return normalised_delta_si


def get_polygons(masks):
    polygons = []
    for mask in masks:
        # mask, header = nrrd.read(mask)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        c = max(contours, key=cv2.contourArea)
        polygons.append(c)

    return masks, polygons


def get_vb_corners(masks):

    vb_masks = masks[0], masks[2]
    corners = []
    for vb_mask in vb_masks:
        vb_mask = cv2.GaussianBlur(vb_mask, (3,3), 0)
        corners_coords = cv2.goodFeaturesToTrack(vb_mask, 20, 0.01, 10)
        corners_coords = np.int0(corners_coords)

        areas = []
        coordinates = []
        for p in itertools.combinations(np.array(corners_coords),4):
            for r in itertools.permutations(p,4):

                array = np.concatenate(r)

                pgon = shapely.Polygon(array)
                area = pgon.area

                coordinates.append(r)
                areas.append(area)

        best_corners = coordinates[areas.index(max(areas))]
        corners.append(best_corners)
        
    return corners


def get_ivd_midline_cv2_fitline(masks, polygons):

    [vx, vy, x, y] = cv2.fitLine(polygons[1], cv2.DIST_L2, 0, 0.01, 0.01)

    ivd_mask = masks[1]

    rows,cols = ivd_mask.shape[:2]
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    line = [(cols,righty), (0,lefty)]

    ivd_cv2_fit_midline = shapely.geometry.LineString(line)

    return ivd_cv2_fit_midline


def midpoint(x0, y0, x1, y1):

    return shapely.Point((x0 + x1)/2, (y0 + y1)/2)


def extend_line(line, factor):
    """
    Extend a line by a factor of 2
    """
    x0, y0, x1, y1 = line.bounds
    mid = midpoint(x0, y0, x1, y1)
    extended_line = shapely.affinity.scale(line, xfact=factor, yfact=factor, origin=mid)
    return extended_line


def get_vb_diameter(corners, ivd_cv2_fit_midline, polygon):

    # get perpendicular distances from vb corners to ivd midline
    # This is needed to identify the upper and lower corners
    # these for loops will give the perpendicular distance from each corner to the IVD midline
    corner_distances = []
    for corner in corners:
        corner_distances.append(ivd_cv2_fit_midline.distance(shapely.Point(corner[0])))

    # Find the index of the largest two distances using np.argpartition()
    # The largest two distances are the two corners that are farthest from the IVD midline, and therefore the upper and lower corners of the adjacent vertebrae
    lower_corner_idx = np.argpartition(corner_distances, -2) # top two distances i.e. last two entries on the index list
    upper_corner_idx = np.argpartition(corner_distances, 2) # bottom two distances i.e. first two entries on the index list

    # Get the two points that are closest to the intersection points
    stacked_corners = np.vstack(corners)
    lower_corners = stacked_corners[lower_corner_idx[:2]] # index from the corners based on the results of the previous block of code
    upper_corners = stacked_corners[upper_corner_idx[-2:]]

    # get the midpoints of every possible two lines connecting the upper and lower corners
    m1 = midpoint(lower_corners[0][0], lower_corners[0][1], upper_corners[0][0], upper_corners[0][1])
    m2 = midpoint(lower_corners[0][0], lower_corners[0][1], upper_corners[1][0], upper_corners[1][1])
    m3 = midpoint(lower_corners[1][0], lower_corners[1][1], upper_corners[0][0], upper_corners[0][1])
    m4 = midpoint(lower_corners[1][0], lower_corners[1][1], upper_corners[1][0], upper_corners[1][1])

    # get the distance between midpoints of every possible two lines connecting the upper and lower corners
    d1 = m1.distance(m2)
    d2 = m3.distance(m4)
    d3 = m1.distance(m3)
    d4 = m2.distance(m4)
    d5 = m1.distance(m4)
    d6 = m2.distance(m3)
    midpoints = [[m1, m2], [m3, m4], [m1, m3], [m2, m4], [m1, m4], [m2, m3]]
    d = [d1, d2, d3, d4, d5, d6]

    # The two that are further apart represent the two lines conencting the upper and lower corners of the vertebra
    # find the index of the largest value for distance
    vb_height_mid_points = midpoints[np.argmax(d)]

    # these midpoints then need to get extended to make sure they will intersect with the VB polygon,
    # and then those intersections can be used to calculate the VB diameter
    vb_midline = shapely.geometry.LineString([vb_height_mid_points[0], vb_height_mid_points[1]])

    # VB midline inclinication:
    x0 = vb_midline.coords[0][0]
    y0 = vb_midline.coords[0][1]
    x1 = vb_midline.coords[1][0]
    y1 = vb_midline.coords[1][1]

    # Find the angle of the line
    theta = math.atan2((y1 - y0), (x1 - x0))
    slope = (y1 - y0)/(x1 - x0)
    theta = math.atan(slope)
    vb_midline_inclination = math.degrees(theta)

    # use the extend line function from above
    extended_vb_midline = extend_line(vb_midline, 2)

    vb_contour = np.squeeze(polygon)
    vb_shapely_poly = shapely.geometry.Polygon(vb_contour)

    # Find the intersection points
    vb_diameter_points = vb_shapely_poly.intersection(extended_vb_midline)

    if vb_diameter_points.geom_type == "MultiLineString":
        # sometimes the midline will cross the VB polygon multiple times because it has curves
        coords = np.asarray([l.coords for l in vb_diameter_points.geoms])
        x0 = coords[:,0][0][0]
        y0 = coords[:,0][0][1]
        x1 = coords[:,-1][-1][0]
        y1 = coords[:,-1][-1][1]
        vb_diameter_points = shapely.geometry.LineString([(x0, y0), (x1, y1)])

    vb_diameter = vb_diameter_points.length

    # return vb_diameter, vb_diameter_points, lower_corners, upper_corners, vb_midline_inclination
    return vb_diameter, vb_midline_inclination


def get_point_on_vector(initial_pt, terminal_pt, distance):

    v = np.array(initial_pt, dtype=float)
    u = np.array(terminal_pt, dtype=float)
    n = v - u
    n /= np.linalg.norm(n, 2)
    point = v - distance * n

    return tuple(point)    


def getExtrapoledLine(p1,p2):

    # Creates a line extrapoled in p1 to p2 direction
    ratio = 2
    a = p1
    b = (p1[0] + ratio * (p2[0] - p1[0]), p1[1] + ratio * (p2[1] - p1[1]) )

    return shapely.geometry.LineString([a,b])


def cut_polygon_by_line(polygon, line):

    merged = linemerge([polygon.boundary, line])
    borders = unary_union(merged)
    polygons = polygonize(borders)

    return list(polygons)


def label_vb_corners(ivd_cv2_fit_midline, polygons, corners):

    contour = np.squeeze(polygons[1])
    shapely_poly = shapely.geometry.Polygon(contour)

    # Find the intersection points between the fit midline and the IVD contour:
    intersection = shapely_poly.intersection(ivd_cv2_fit_midline)
    if intersection.geom_type == "LineString":
        x0 = intersection.coords[0][0]
        y0 = intersection.coords[0][1]
        x1 = intersection.coords[1][0]
        y1 = intersection.coords[1][1]
    elif intersection.geom_type == "MultiLineString":
        # sometimes the midline will cross the IVD polygon multiple times because it has curves
        coords = np.asarray([l.coords for l in intersection.geoms])
        x0 = coords[:,0][0][0]
        y0 = coords[:,0][0][1]
        x1 = coords[:,-1][-1][0]
        y1 = coords[:,-1][-1][1] #taking only the first and the last coordinates of the intersection line and ignoring any intermediate ones that might get picked up
    elif intersection.geom_type == "GeometryCollection":
        # sometimes there is a single point appended at the end and we need to skip it
        list_coords = []
        for geom in intersection.geoms:
            coords = np.asarray([geom.coords])
            list_coords.append(coords)
        x0 = list_coords[0][:,0][0][0]
        y0 = list_coords[0][:,0][0][1]
        x1 = list_coords[-2][:,-1][-1][0]
        y1 = list_coords[-2][:,-1][-1][1]
        intersection = shapely.geometry.LineString([(x0, y0), (x1, y1)])
    
    # Find points in corners that are closest to the intersection points
    # Get the distance between the intersection points and the corners
    # x0, y0 are the anterior IVD midline intersection coordinates, x1, y1 are the posterior IVD midline intersection coordinates
    corners = np.vstack(corners)

    anterior = [] # Not necessarily anterior or posterior, but at least separated 
    anterior_ivd = (x0, y0)
    for c in corners:
        corner = c[0]
        distance = math.dist(anterior_ivd, corner)
        anterior.append(distance)

    posterior = []
    posterior_ivd = (x1, y1)
    for c in corners:
        corner = c[0]
        distance = math.dist(posterior_ivd, corner)
        posterior.append(distance)

    # Find the index of the minimum two distances using np.argpartition()
    smallest_anterior_idx = np.argpartition(anterior, 2)
    smallest_posterior_idx = np.argpartition(posterior, 2)

    # Get the two points that are closest to the intersection points
    two_anterior_corners = corners[smallest_anterior_idx[:2]]
    two_posterior_corners = corners[smallest_posterior_idx[:2]]

    return two_anterior_corners, two_posterior_corners


def get_ivd_midline_vb_corners(ivd_cv2_fit_midline, polygons, corners):

    two_anterior_corners, two_posterior_corners = label_vb_corners(ivd_cv2_fit_midline, polygons, corners)

    anterior_midpoint = midpoint(two_anterior_corners[0][0][0],
                                 two_anterior_corners[0][0][1],
                                 two_anterior_corners[1][0][0],
                                 two_anterior_corners[1][0][1])
    posterior_midpoint = midpoint(two_posterior_corners[0][0][0],
                                 two_posterior_corners[0][0][1],
                                 two_posterior_corners[1][0][0],
                                 two_posterior_corners[1][0][1])
    
    # Create a line between the two midpoints
    ivd_midline = shapely.geometry.LineString([anterior_midpoint, posterior_midpoint])

    extended_ivd_midline = extend_line(ivd_midline, 2)

    # Convert the cv2 contour to a shapely polygon
    contour = np.squeeze(polygons[1])
    shapely_poly = shapely.geometry.Polygon(contour)

    # Find the intersection points
    intersections = shapely_poly.intersection(extended_ivd_midline)

    if intersections.geom_type == "MultiLineString":
        # sometimes the midline will cross the IVD polygon multiple times because it has curves
        coords = np.asarray([l.coords for l in intersections.geoms])
        x0 = coords[:,0][0][0]
        y0 = coords[:,0][0][1]
        x1 = coords[:,-1][-1][0]
        y1 = coords[:,-1][-1][1]
        intersections = shapely.geometry.LineString([(x0, y0), (x1, y1)])
    
    if intersections.geom_type == "GeometryCollection":
        # sometimes there is a single point appended at the end and we need to skip it
        list_coords = []
        for geom in intersections.geoms:
            coords = np.asarray([geom.coords])
            list_coords.append(coords)
        x0 = list_coords[0][:,0][0][0]
        y0 = list_coords[0][:,0][0][1]
        x1 = list_coords[-2][:,-1][-1][0]
        y1 = list_coords[-2][:,-1][-1][1]
        intersections = shapely.geometry.LineString([(x0, y0), (x1, y1)])

    disc_diameter = math.sqrt((intersections.coords[1][0] - intersections.coords[0][0])**2 + (intersections.coords[1][1] - intersections.coords[0][1])**2)

    # return extended_ivd_midline, intersections, anterior_midpoint, posterior_midpoint, disc_diameter
    return intersections, disc_diameter


def cut_polygon_by_lines(intersections, polygons, mu):
    """
    Based on percentage value mu, cut the polygon into two parts
    """
    x0 = intersections.coords[0][0]
    y0 = intersections.coords[0][1]
    x1 = intersections.coords[1][0]
    y1 = intersections.coords[1][1]

    # Find the angle of the line
    theta = math.atan2((y1 - y0), (x1 - x0))
    slope = (y1 - y0)/(x1 - x0)
    theta = math.atan(slope)
    ivd_inclination = math.degrees(theta)

    # Find the points that contain the central mu% of the line
    # Find the distance between the points
    IVD_length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    distance = (IVD_length - (IVD_length * mu)) / 2
    start_point = (x0, y0)
    end_point = (x1, y1)

    point_a = get_point_on_vector(start_point, end_point, distance)
    point_b = get_point_on_vector(end_point, start_point, distance)
    point_c = midpoint(x0, y0, x1, y1) # for getting the central point of the IVD

    # Find the perpendicular lines
    if slope == 0:
        angle =  math.pi / 2
    else:
        perp_slope = -1.0 / slope
        angle = math.atan(perp_slope)

    length = 30
    start = shapely.geometry.Point(point_a[0] - length * math.cos(angle),
                                point_a[1] - length * math.sin(angle))
    end = shapely.geometry.Point(point_a[0] + length * math.cos(angle),
                                point_a[1] + length * math.sin(angle))
    line_a = shapely.geometry.LineString([start, end])

    start = shapely.geometry.Point(point_b[0] - length * math.cos(angle),
                                point_b[1] - length * math.sin(angle))
    end = shapely.geometry.Point(point_b[0] + length * math.cos(angle),
                                point_b[1] + length * math.sin(angle))
    line_b = shapely.geometry.LineString([start, end])

    start = shapely.geometry.Point(point_c.x - length * math.cos(angle),
                                point_c.y - length * math.sin(angle))
    end = shapely.geometry.Point(point_c.x + length * math.cos(angle),
                                point_c.y + length * math.sin(angle))
    line_c = shapely.geometry.LineString([start, end])

    # Convert the cv2 contour to a shapely polygon
    contour = np.squeeze(polygons[1])
    shapely_poly = shapely.geometry.Polygon(contour)

    # intersections_a = shapely_poly.intersection(line_a)
    # intersections_b = shapely_poly.intersection(line_b)
    short_midline_intersections = shapely_poly.intersection(line_c)
    if short_midline_intersections.geom_type == "MultiLineString":
        # sometimes the midline will cross the VB polygon multiple times because it has curves
        coords = np.asarray([l.coords for l in short_midline_intersections.geoms])
        x0 = coords[:,0][0][0]
        y0 = coords[:,0][0][1]
        x1 = coords[:,-1][-1][0]
        y1 = coords[:,-1][-1][1]
        short_midline_intersections = shapely.geometry.LineString([(x0, y0), (x1, y1)])

    cut_1 = cut_polygon_by_line(shapely_poly, line_a)

    # sometimes with an unusually shaped disc the lines splitting the disc will result in 3 or 4 polygons, not two. In this case, return the two polygons with the largest area
    if len(cut_1) > 2:
        areas = [p.area for p in cut_1]
        largest_area = max(areas)
        largest_area_index = areas.index(largest_area)
        polygons_a = [cut_1[largest_area_index]]

        # remove the largest area from the list
        areas.pop(largest_area_index)
        largest_area = max(areas)
        largest_area_index = areas.index(largest_area)
        polygons_a.append(cut_1[largest_area_index])
    
    else:   
        polygons_a = cut_1

    cut_2 = cut_polygon_by_line(shapely_poly, line_b)

    # if polygon_b length is greater than 2, return the two polygons with the largest area
    if len(cut_2) > 2:
        areas = [p.area for p in cut_2]
        largest_area = max(areas)
        largest_area_index = areas.index(largest_area)
        polygons_b = [cut_2[largest_area_index]]

        # remove the largest area from the list
        areas.pop(largest_area_index)
        largest_area = max(areas)
        largest_area_index = areas.index(largest_area)
        polygons_b.append(cut_2[largest_area_index])
    
    else:
        polygons_b = cut_2

    polygons = polygons_a + polygons_b

    return polygons, ivd_inclination


def get_mid_polygon(polygons):
    """
    Return the polygon that covers the centre of the IVD
    """
    indexes = [0, 1, 2, 3]
    pairs = [(a, b) for idx, a in enumerate(indexes) for b in indexes[idx + 1:]]

    intersections = []
    for pair in pairs:
        intersection = polygons[pair[0]].intersection(polygons[pair[1]])
        intersections.append(intersection)
    ivd_centre = max(intersections, key=lambda x: x.area)
    # get area of disc centre

    return ivd_centre


def get_areas(polygons, ivd_centre):
    
    central_ivd_area = ivd_centre.area

    contour = np.squeeze(polygons[1])
    ivd_poly = shapely.geometry.Polygon(contour)
    total_ivd_area = ivd_poly.area

    contour = np.squeeze(polygons[0])
    upper_vb_poly = shapely.geometry.Polygon(contour)
    upper_vb_area = upper_vb_poly.area

    contour = np.squeeze(polygons[2])
    lower_vb_poly = shapely.geometry.Polygon(contour)
    lower_vb_area = lower_vb_poly.area

    return central_ivd_area, total_ivd_area, upper_vb_area, lower_vb_area


def calculate_indices(fsu, image):

    cranial_vb_mask, _ = nrrd.read(fsu[0])
    ivd_mask, _ = nrrd.read(fsu[1])
    caudal_vb_mask, _ = nrrd.read(fsu[2])
    fsu = [cranial_vb_mask, ivd_mask, caudal_vb_mask]
    image, _ = nrrd.read(image) 

    fsu, polygons = get_polygons(fsu)

    normalised_delta_si = get_delta_si(fsu, image)

    corners = get_vb_corners(fsu)

    # IVD midline based on CV2 fit
    ivd_cv2_fit_midline = get_ivd_midline_cv2_fitline(fsu, polygons)

    cranial_vb_diameter, \
    cranial_vb_midline_inclination = get_vb_diameter(corners[0], ivd_cv2_fit_midline, polygons[0])

    caudal_vb_diameter, \
    caudal_vb_midline_inclination = get_vb_diameter(corners[1], ivd_cv2_fit_midline, polygons[2])

    # IVD midline based on VB corners 
    intersections, \
    disc_diameter = get_ivd_midline_vb_corners(ivd_cv2_fit_midline, polygons, corners)

    # Cutting the IVD into 3 parts based on mu percentage
    ivd_segmentation_polygons, \
    ivd_inclination = cut_polygon_by_lines(intersections, polygons = polygons, mu = 0.8)

    # Get the middle part of the IVD
    ivd_centre = get_mid_polygon(ivd_segmentation_polygons)

    # Calculate the areas
    central_ivd_area, \
    ivd_area, \
    cranial_vb_area, \
    caudal_vb_area = get_areas(polygons, ivd_centre)

    # Final calculation of indices
    cranial_vb_height = cranial_vb_area / cranial_vb_diameter
    caudal_vb_height = caudal_vb_area / caudal_vb_diameter
    bulge_area = ivd_area - central_ivd_area
    bulge_index = bulge_area / ivd_area
    ivd_height = central_ivd_area / disc_diameter
    ivd_height_index = 2 * ivd_height / (cranial_vb_height + caudal_vb_height)

    results = {
        'cranial_vb_diameter': cranial_vb_diameter,
        'cranial_vb_midline_inclination': cranial_vb_midline_inclination,
        'caudal_vb_diameter': caudal_vb_diameter,
        'caudal_vb_midline_inclination': caudal_vb_midline_inclination,
        'disc_diameter': disc_diameter,
        'ivd_inclination': ivd_inclination,
        'central_ivd_area': central_ivd_area,
        'ivd_area': ivd_area,
        'cranial_vb_area': cranial_vb_area,
        'caudal_vb_area': caudal_vb_area,
        'cranial_vb_height': cranial_vb_height,
        'caudal_vb_height': caudal_vb_height,
        'bulge_area': bulge_area,
        'bulge_index': bulge_index,
        'ivd_height': ivd_height,
        'ivd_height_index': ivd_height_index
        'normalised_delta_si': normalised_delta_si
    }

    return results
